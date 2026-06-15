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
import time

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
# In-cluster embedding served by the Slemify-trained retriever (task:
# embedding) over a TEI-compatible /embed endpoint. The domain-tuned encoder
# produces 768-dimensional vectors and must match the dimension used at index
# time in index-knowledge.py.
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:8083")
# In-cluster cross-encoder re-ranker. Scores the OpenSearch candidate set and
# keeps only the most relevant chunks, so the auditor prompt stays small.
RERANKER_URL = os.environ.get("RERANKER_URL", "http://localhost:8084")
INDEX_NAME = os.environ.get("INDEX_NAME", "k8s-autoscaling-knowledge")
# Candidates pulled from vector search before re-ranking. The cross-encoder
# reranker scores every candidate on CPU, so this count is the main lever on
# rerank latency (~linear in candidates). The tuned retriever already ranks
# well, so 6 candidates keep nearly all the recall of 10 while cutting rerank
# time by ~40%. The orchestrator keeps only the top few of these for the SLM.
RETRIEVE_CANDIDATES = 6
# Query budget (chars) sent to the cross-encoder. The reranker only needs the
# question's intent to score doc relevance, and it re-encodes the query against
# every candidate. Long config pastes blow up rerank latency, so we cap the
# query here (the full text still goes to the auditor SLM untouched).
RERANK_QUERY_CHARS = 512

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
    """Embed text via the Slemify-trained retriever (TEI /embed, 768d)."""
    with httpx.Client(timeout=10) as client:
        resp = client.post(f"{EMBEDDING_URL}/embed", json={"inputs": text[:8000]})
        resp.raise_for_status()
        # TEI returns a list of embeddings, one per input.
        return resp.json()[0]


def rerank_docs(query: str, docs: list[str], top_k: int) -> list[str]:
    """Re-rank candidate docs with the cross-encoder, keeping the best top_k.

    Falls back to the original order (truncated) if the reranker is unavailable,
    so retrieval still works even if the reranker pod is down.
    """
    if not docs:
        return []
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{RERANKER_URL}/rerank",
                json={"query": query[:RERANK_QUERY_CHARS], "documents": docs, "top_k": top_k},
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        return [docs[r["index"]] for r in results]
    except Exception as e:
        print(f"  Rerank failed, using vector order: {e}")
        return docs[:top_k]


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

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, rerank_docs, "warmup", ["warmup document"], 1)
        print("  Warmup reranker: ok")
    except Exception as e:
        print(f"  Warmup reranker: failed ({e})")

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


def vector_search(embedding: list[float], k: int) -> list[str]:
    """k-NN search over the indexed corpus for a precomputed query embedding."""
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
        for hit in results["hits"]["hits"]
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
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        def elapsed_ms(t0: float) -> int:
            return round((time.perf_counter() - t0) * 1000)

        # --- Step 1: Triage SLM ---
        yield sse("step_start", name="Triage SLM (4B, CPU)", note="classifying intent")
        t = time.perf_counter()
        result = await loop.run_in_executor(None, classify, q.text)
        yield sse(
            "step_done", name="Triage SLM (4B, CPU)", ms=elapsed_ms(t),
            detail=f"{result['category'].replace('_', ' ')} · {result['confidence']} confidence",
        )

        # Route noise (no retrieval, no generation)
        if result["category"] == "noise":
            yield sse("response", text="This does not look like a K8s autoscaling question.")
            yield sse("total", ms=elapsed_ms(t_start))
            yield "data: [DONE]\n\n"
            return

        low_conf = result["confidence"] in ("low", "unknown")
        keep_k = 5 if low_conf else 2

        # --- Step 2: Embed query (Slemify-tuned retriever) ---
        yield sse("step_start", name="Retriever (tuned encoder, CPU)", note="embedding query → 768d")
        t = time.perf_counter()
        embedding = await loop.run_in_executor(None, embed_query, q.text)
        yield sse("step_done", name="Retriever (tuned encoder, CPU)", ms=elapsed_ms(t),
                  detail="domain-tuned ONNX encoder")

        # --- Step 3: Vector search (OpenSearch k-NN) ---
        yield sse("step_start", name="OpenSearch (vector DB)", note=f"k-NN search, top {RETRIEVE_CANDIDATES}")
        t = time.perf_counter()
        candidates = await loop.run_in_executor(None, vector_search, embedding, RETRIEVE_CANDIDATES)
        yield sse("step_done", name="OpenSearch (vector DB)", ms=elapsed_ms(t),
                  detail=f"{len(candidates)} candidate chunks")

        # --- Step 4: Rerank (cross-encoder) ---
        yield sse("step_start", name="Reranker (cross-encoder, CPU)",
                  note=f"scoring {len(candidates)} pairs → top {keep_k}")
        t = time.perf_counter()
        docs = await loop.run_in_executor(None, rerank_docs, q.text, candidates, keep_k)
        yield sse("step_done", name="Reranker (cross-encoder, CPU)", ms=elapsed_ms(t),
                  detail=f"kept top {len(docs)}")

        context = "\n\n---\n\n".join(docs)

        # --- Step 5: Generation (Auditor SLM, or LLM fallback on low confidence) ---
        if low_conf:
            gen_name = "LLM API (Bedrock fallback)"
            stream_fn = stream_llm
            yield sse("model", name="Claude Sonnet 4.5 (Bedrock)")
        else:
            gen_name = "Auditor SLM (8B, CPU)"
            stream_fn = stream_slm
            yield sse("model", name="Auditor SLM (8B, CPU)")

        yield sse("step_start", name=gen_name, note="generating answer")
        t = time.perf_counter()
        first = True
        async for token in stream_fn(q.text, context):
            if first:
                # Report time-to-first-token: the latency the user actually
                # waits before the answer starts streaming.
                yield sse("step_done", name=gen_name, ms=elapsed_ms(t),
                          detail="time to first token")
                first = False
            yield sse("token", text=token)
        if first:
            yield sse("step_done", name=gen_name, ms=elapsed_ms(t), detail="no output")

        yield sse("total", ms=elapsed_ms(t_start))
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
.msg.step{align-self:stretch;max-width:100%;display:flex;align-items:center;gap:10px;background:var(--surface);border:1px solid var(--border);padding:8px 12px;border-radius:8px;font-size:13px}
.msg.step .step-name{font-weight:600}
.msg.step .step-note{color:var(--muted);font-size:12px;flex:1}
.msg.step .dur{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:600;color:var(--accent)}
.msg.step.done .dur{color:var(--slm-color)}
.msg.step .spinner{width:12px;height:12px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;display:inline-block;flex-shrink:0}
.msg.step .check{color:var(--slm-color);font-weight:700;flex-shrink:0}
.msg.total{align-self:stretch;max-width:100%;text-align:right;color:var(--strong);font-weight:600;font-size:13px;padding:4px 12px;border-top:1px solid var(--border)}
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
@keyframes spin{to{transform:rotate(360deg)}}
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

let steps = {};
function fmt(ms) { return ms >= 1000 ? (ms/1000).toFixed(2) + 's' : ms + ' ms'; }

function escapeHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function addStep(name, note) {
  const div = document.createElement('div');
  div.className = 'msg step running';
  div.innerHTML = '<span class="spinner"></span>' +
    '<span class="step-name">' + escapeHtml(name) + '</span>' +
    '<span class="step-note">' + escapeHtml(note || '') + '</span>' +
    '<span class="dur"></span>';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  steps[name] = div;
}

function finishStep(name, ms, detail) {
  const div = steps[name];
  if (!div) return;
  div.className = 'msg step done';
  const spinner = div.querySelector('.spinner');
  if (spinner) { const c = document.createElement('span'); c.className = 'check'; c.textContent = '\u2713'; spinner.replaceWith(c); }
  if (detail) div.querySelector('.step-note').textContent = detail;
  div.querySelector('.dur').textContent = fmt(ms);
}

function clearChat() { chat.innerHTML = ''; }

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  btn.disabled = true;
  steps = {};
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
        if (msg.type === 'step_start') {
          addStep(msg.name, msg.note);
        } else if (msg.type === 'step_done') {
          finishStep(msg.name, msg.ms, msg.detail);
        } else if (msg.type === 'total') {
          addMsg('Total pipeline time: <strong>' + fmt(msg.ms) + '</strong>', 'total');
        } else if (msg.type === 'model') {
          const isSlm = msg.name.toLowerCase().includes('slm');
          addMsg(msg.name, 'model-badge ' + (isSlm ? 'slm' : 'llm'));
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
