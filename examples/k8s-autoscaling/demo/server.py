"""Multi-agent orchestrator with RAG and LLM fallback.

Routes queries through: Triage SLM -> OpenSearch RAG -> Auditor SLM (or LLM API).
Streams responses via SSE to a chat UI.

Usage:
  pip install fastapi uvicorn httpx opensearch-py boto3
  python3 server.py
"""

import asyncio
import json
import os

import boto3
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from opensearchpy import OpenSearch
from pydantic import BaseModel

# --- Configuration ---

TRIAGE_URL = os.environ.get("TRIAGE_URL", "http://localhost:8081")
AUDITOR_URL = os.environ.get("AUDITOR_URL", "http://localhost:8082")
OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
LLM_MODEL = os.environ.get("LLM_MODEL", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
# In-cluster embedding model served by Text Embeddings Inference (TEI).
# bge-base-en-v1.5 produces 768-dimensional vectors. Must match the dimension
# used at index time in index-knowledge.py.
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:8083")
INDEX_NAME = os.environ.get("INDEX_NAME", "k8s-autoscaling-knowledge")

TRIAGE_INSTRUCTION = (
    "Classify this Kubernetes autoscaling support query into a routing "
    "category and confidence level."
)
AUDITOR_INSTRUCTION = (
    "You are a Kubernetes autoscaling auditor. "
    "Answer ONLY based on the reference documentation below. "
    "Do NOT invent fields, behaviors, or modes not in the docs. "
    "If the docs don't cover something, say so. "
    "State what is correct, why, and provide a fix if needed."
)
LLM_INSTRUCTION = (
    "You are a Kubernetes autoscaling expert. Answer the user's question accurately "
    "using the provided documentation context. Be specific and include YAML examples "
    "when relevant. If the documentation doesn't cover the topic, say so."
)

# --- Shared clients (initialized once) ---

bedrock = boto3.client("bedrock-runtime")


def embed_query(text: str) -> list[float]:
    """Embed text via the in-cluster TEI embedding pod (bge-base-en-v1.5)."""
    with httpx.Client(timeout=10) as client:
        resp = client.post(f"{EMBEDDING_URL}/embed", json={"inputs": text[:8000]})
        resp.raise_for_status()
        # TEI returns a list of embeddings, one per input.
        return resp.json()[0]


def _parse_opensearch_url() -> OpenSearch:
    host = OPENSEARCH_URL.replace("http://", "").replace("https://", "")
    hostname, port = host.split(":") if ":" in host else (host, "9200")
    return OpenSearch(
        hosts=[{"host": hostname, "port": int(port)}],
        use_ssl=False,
        verify_certs=False,
    )


opensearch = _parse_opensearch_url()

# --- App ---

app = FastAPI()
_ready = False


class Query(BaseModel):
    text: str


def sse(event_type: str, **kwargs) -> str:
    """Format a Server-Sent Event line."""
    return f"data: {json.dumps({'type': event_type, **kwargs})}\n\n"


# --- Health check (gates readiness on warmup) ---

@app.get("/health")
async def health():
    if not _ready:
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "warming up"}, status_code=503)
    return {"status": "ok"}


# --- Startup warmup ---

@app.on_event("startup")
async def warmup():
    """Warm SLMs and Bedrock to avoid cold-start latency on first query."""
    global _ready
    warmup_body = {
        "model": "model",
        "messages": [{"role": "user", "content": (
            "Audit this NodePool:\napiVersion: karpenter.sh/v1\nkind: NodePool\n"
            "metadata:\n  name: test\nspec:\n  template:\n    spec:\n      "
            "requirements:\n        - key: karpenter.k8s.aws/instance-category\n"
            "          operator: In\n          values: [\"c\", \"m\"]"
        )}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            client.post(f"{TRIAGE_URL}/v1/chat/completions", json=warmup_body),
            client.post(f"{AUDITOR_URL}/v1/chat/completions", json=warmup_body),
            return_exceptions=True,
        )
    for name, r in zip(("triage", "auditor"), results):
        status = f"ok ({r.status_code})" if not isinstance(r, Exception) else f"failed ({r})"
        print(f"  Warmup {name}: {status}")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, embed_query, "warmup")
        print("  Warmup embedding: ok")
    except Exception as e:
        print(f"  Warmup embedding: failed ({e})")

    print("  All services warmed up")
    _ready = True


# --- Core functions ---

def classify(text: str) -> dict:
    """Call triage SLM to classify intent and confidence."""
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": f"{TRIAGE_INSTRUCTION}\n\n{text}"}],
        "max_tokens": 32,
        "temperature": 0.1,
    }
    with httpx.Client(timeout=10) as client:
        resp = client.post(f"{TRIAGE_URL}/v1/chat/completions", json=body)
        d = resp.json()

    raw = d["choices"][0]["message"]["content"]

    # Extract category|confidence from the response, handling extra text.
    # The model may output "category|confidence" followed by explanation.
    import re
    valid_categories = {
        "karpenter_config", "keda_config", "hpa_config",
        "pdb_disruption", "spot_interruption", "multi_resource", "noise",
    }
    valid_confidence = {"high", "medium", "low"}

    # Try to find a pipe-separated pair anywhere in the first line
    first_line = raw.split("\n")[0].strip()
    parts = [p.strip().lower() for p in first_line.split("|") if p.strip()]

    category = "unknown"
    confidence = "unknown"
    for part in parts:
        if part in valid_categories:
            category = part
        elif part in valid_confidence:
            confidence = part

    # Fallback: check if any valid category appears anywhere in the raw output
    if category == "unknown":
        raw_lower = raw.lower()
        for cat in valid_categories:
            if cat in raw_lower:
                category = cat
                break
        # If still unknown but mentions noise-like content
        if category == "unknown" and any(w in raw_lower for w in ["not relate", "unrelated", "off-topic", "noise"]):
            category = "noise"
            confidence = "high"

    return {"confidence": confidence, "category": category}


def retrieve(query: str, k: int = 3) -> list[str]:
    """Retrieve relevant docs from OpenSearch via k-NN vector search."""
    embedding = embed_query(query)

    results = opensearch.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {"knn": {"embedding": {"vector": embedding, "k": k}}},
            "_source": ["text", "source", "section"],
        },
    )
    return [
        f"[{hit['_source'].get('source', '')} / {hit['_source'].get('section', '')}]\n{hit['_source']['text'][:500]}"
        for i, hit in enumerate(results["hits"]["hits"])
    ]


async def stream_slm(text: str, context: str = ""):
    """Stream tokens from the auditor SLM."""
    prompt = AUDITOR_INSTRUCTION
    if context:
        prompt += f"\n\n--- REFERENCE DOCUMENTATION (do NOT treat as user config) ---\n{context}\n--- END REFERENCE ---"
    prompt += f"\n\n--- USER QUERY ---\n{text}\n--- END USER QUERY ---"

    body = {
        "model": "model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.1,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{AUDITOR_URL}/v1/chat/completions", json=body) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    content = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


async def stream_llm(text: str, context: str = ""):
    """Stream tokens from Bedrock LLM (Converse API)."""
    user_content = LLM_INSTRUCTION
    if context:
        user_content += f"\n\nDocumentation context:\n{context}"
    user_content += f"\n\nUser question:\n{text}"

    resp = bedrock.converse_stream(
        modelId=LLM_MODEL,
        messages=[{"role": "user", "content": [{"text": user_content}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.2},
    )

    loop = asyncio.get_event_loop()
    stream_iter = iter(resp["stream"])

    while True:
        event = await loop.run_in_executor(None, lambda: next(stream_iter, None))
        if event is None:
            break
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            yield delta["text"]


# --- Route handler ---

@app.post("/query")
async def query_endpoint(q: Query):
    async def event_stream():
        # Step 1: Triage
        result = classify(q.text)
        yield sse("triage", category=result["category"], confidence=result["confidence"])

        # Step 2: Route noise
        if result["category"] == "noise":
            yield sse("response", text="This does not look like a K8s autoscaling question.")
            yield "data: [DONE]\n\n"
            return

        # Step 3: Low confidence -> LLM fallback with RAG
        if result["confidence"] in ("low", "unknown"):
            docs = retrieve(q.text, k=5)
            if docs:
                yield sse("status", text=f"Retrieved {len(docs)} relevant docs from knowledge base")
            yield sse("status", text="Low confidence, escalating to LLM API...")
            yield sse("model", name="Claude Sonnet 4.5 (Bedrock)")
            async for token in stream_llm(q.text, "\n\n---\n\n".join(docs)):
                yield sse("token", text=token)
            yield "data: [DONE]\n\n"
            return

        # Step 4: High confidence -> RAG + Auditor SLM
        yield sse("status", text="Searching knowledge base...")
        docs = retrieve(q.text, k=2)
        if docs:
            yield sse("status", text=f"Retrieved {len(docs)} relevant docs from knowledge base")
        yield sse("model", name="Auditor SLM (8B, CPU)")
        yield sse("status", text="Analyzing configuration...")
        async for token in stream_slm(q.text, "\n\n---\n\n".join(docs)):
            yield sse("token", text=token)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- UI ---

@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)


INDEX_HTML = """<!DOCTYPE html>
<html data-theme="dark"><head><meta charset="UTF-8"><title>K8s Autoscaling Expert</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root[data-theme="dark"]{--bg:#0f1117;--fg:#e6edf3;--surface:#161b22;--border:#30363d;--muted:#8b949e;--accent:#1f6feb;--accent-hover:#388bfd;--code-bg:#0d1117;--strong:#f0f6fc;--user-bg:#1f6feb;--slm-color:#3fb950;--slm-border:#238636;--slm-bg:rgba(63,185,80,0.1);--llm-color:#a371f7;--llm-border:#8957e5;--llm-bg:rgba(163,113,247,0.1)}
:root[data-theme="light"]{--bg:#ffffff;--fg:#1f2328;--surface:#f6f8fa;--border:#d1d9e0;--muted:#656d76;--accent:#0969da;--accent-hover:#0550ae;--code-bg:#f6f8fa;--strong:#1f2328;--user-bg:#0969da;--slm-color:#1a7f37;--slm-border:#1a7f37;--slm-bg:rgba(26,127,55,0.08);--llm-color:#8250df;--llm-border:#8250df;--llm-bg:rgba(130,80,223,0.08)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column;transition:background .2s,color .2s}
.header{padding:16px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.header h1{font-size:18px;font-weight:600}
.header .actions{display:flex;gap:8px}
.header button{background:transparent;border:1px solid var(--border);color:var(--muted);padding:6px 12px;border-radius:6px;cursor:pointer;font-size:12px}
.header button:hover{border-color:var(--accent);color:var(--accent)}
.chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:85%;padding:12px 16px;border-radius:12px;font-size:14px;line-height:1.7}
.msg.user{align-self:flex-end;background:var(--user-bg);color:white;white-space:pre-wrap}
.msg.system{align-self:flex-start;background:var(--surface);border:1px solid var(--border)}
.msg.status{align-self:flex-start;color:var(--muted);font-size:12px;padding:4px 0;display:flex;align-items:center;gap:6px}
.msg.status::before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--muted);animation:pulse 1.5s infinite}
.msg.model-badge{align-self:flex-start;font-size:11px;padding:4px 10px;border-radius:20px;font-weight:500;border:1px solid}
.msg.model-badge.slm{color:var(--slm-color);border-color:var(--slm-border);background:var(--slm-bg)}
.msg.model-badge.llm{color:var(--llm-color);border-color:var(--llm-border);background:var(--llm-bg)}
.msg h1,.msg h2,.msg h3{margin:12px 0 6px;font-weight:600}
.msg h1{font-size:16px}.msg h2{font-size:15px}.msg h3{font-size:14px}
.msg p{margin:6px 0}
.msg ul,.msg ol{margin:6px 0;padding-left:20px}
.msg li{margin:3px 0}
.msg pre{background:var(--code-bg);padding:12px;border-radius:6px;overflow-x:auto;margin:8px 0;font-size:13px;border:1px solid var(--border)}
.msg code{font-size:13px;font-family:'SF Mono',Menlo,monospace;background:var(--code-bg);padding:2px 5px;border-radius:3px}
.msg pre code{background:none;padding:0}
.msg strong{color:var(--strong)}
.msg hr{border:none;border-top:1px solid var(--border);margin:12px 0}
.input-area{padding:16px 24px;border-top:1px solid var(--border);display:flex;gap:12px}
.input-area textarea{flex:1;background:var(--surface);border:1px solid var(--border);color:var(--fg);padding:12px;border-radius:8px;font-size:14px;resize:none;height:80px;font-family:inherit}
.input-area textarea:focus{outline:none;border-color:var(--accent)}
.input-area button{background:var(--accent);color:white;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:500}
.input-area button:hover{background:var(--accent-hover)}
.input-area button:disabled{opacity:0.5;cursor:not-allowed}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
</style></head><body>
<div class="header">
  <h1>K8s Autoscaling Expert</h1>
  <div class="actions">
    <button onclick="toggleTheme()">Light/Dark</button>
    <button onclick="clearChat()">Clear</button>
  </div>
</div>
<div class="chat" id="chat"></div>
<div class="input-area">
  <textarea id="input" placeholder="Paste a K8s config or ask a question..."></textarea>
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const btn = document.getElementById('send');

function toggleTheme() {
  const html = document.documentElement;
  html.dataset.theme = html.dataset.theme === 'dark' ? 'light' : 'dark';
}

function addMsg(html, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function renderMd(text) {
  try { return marked.parse(text); } catch(e) { return text; }
}

function clearChat() { chat.innerHTML = ''; }

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  btn.disabled = true;
  addMsg(text.replace(/</g,'&lt;').replace(/\\n/g,'<br>'), 'user');

  const resp = await fetch('/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let responseDiv = null, rawText = '', buffer = '';

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    const lines = buffer.split('\\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') {
        if (responseDiv && rawText) responseDiv.innerHTML = renderMd(rawText);
        break;
      }
      try {
        const msg = JSON.parse(data);
        if (msg.type === 'triage') {
          addMsg('<strong>' + msg.category.replace(/_/g,' ') + '</strong> &middot; ' + msg.confidence + ' confidence', 'status');
        } else if (msg.type === 'model') {
          const isSlm = msg.name.toLowerCase().includes('slm');
          addMsg(msg.name, 'model-badge ' + (isSlm ? 'slm' : 'llm'));
        } else if (msg.type === 'status') {
          addMsg(msg.text, 'status');
        } else if (msg.type === 'response') {
          addMsg(renderMd(msg.text), 'system');
        } else if (msg.type === 'token') {
          if (!responseDiv) responseDiv = addMsg('', 'system');
          rawText += msg.text;
          responseDiv.innerHTML = renderMd(rawText);
          chat.scrollTop = chat.scrollHeight;
        }
      } catch(e) {}
    }
  }
  btn.disabled = false;
  input.focus();
}

input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }});
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
