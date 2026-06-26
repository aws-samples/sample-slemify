"""FastAPI app for the k8s-autoscaling agent.

Thin transport layer: warmup, the SSE /query stream (driven by the LangGraph
agent in the `agent` package), the gated /apply remediation endpoint, and the
UI. All orchestration logic lives in `agent/` (config, retrieval, generation,
gate, tools, remediation, classify, extract, graph).

Usage:
  pip install fastapi uvicorn httpx opensearch-py boto3 langgraph kubernetes pyyaml
  uvicorn server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import os
import time

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agent import config, extract, remediation, retrieval, tools
from agent import toolclient
from agent.graph import agent

app = FastAPI()
_ready = False

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "ui.html")) as _f:
    INDEX_HTML = _f.read()


class Query(BaseModel):
    text: str
    autopilot: bool = False


class ApplyRequest(BaseModel):
    action: str
    target: str


def sse(event_type: str, **kwargs) -> str:
    return f"data: {json.dumps({'type': event_type, **kwargs})}\n\n"


@app.get("/health")
async def health():
    if not _ready:
        return JSONResponse({"status": "warming up"}, status_code=503)
    return {"status": "ok"}


@app.on_event("startup")
async def warmup():
    """Warm the SLMs and retrieval to avoid cold-start latency on first query."""
    global _ready
    # When sandboxed (TOOLSVC_URL set), the orchestrator holds no cluster creds;
    # the tools pod runs init_k8s itself. Only init in-process for single-pod dev.
    if not config.TOOLSVC_URL:
        tools.init_k8s()
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            client.post(f"{config.TRIAGE_URL}/v1/chat/completions", json=body),
            client.post(f"{config.AUDITOR_URL}/v1/chat/completions", json=body),
            return_exceptions=True,
        )
    for name, r in zip(("triage", "auditor"), results):
        print(f"  Warmup {name}: {'ok' if not isinstance(r, Exception) else f'failed ({r})'}")
    loop = asyncio.get_event_loop()
    for name, fn in (("embedding", lambda: retrieval.embed_query("warmup")),
                     ("reranker", lambda: retrieval.rerank_docs("warmup", ["warmup doc"], 1))):
        try:
            await loop.run_in_executor(None, fn)
            print(f"  Warmup {name}: ok")
        except Exception as e:
            print(f"  Warmup {name}: failed ({e})")
    print("  All services warmed up")
    _ready = True


@app.post("/query")
async def query_endpoint(q: Query):
    async def event_stream():
        t0 = time.perf_counter()
        # stream_mode="custom" yields exactly the dicts each node writes, so the
        # UI's SSE contract is preserved without LangChain message plumbing.
        async for event in agent.astream({"query": q.text, "autopilot": q.autopilot}, stream_mode="custom"):
            yield f"data: {json.dumps(event)}\n\n"
        yield sse("total", ms=round((time.perf_counter() - t0) * 1000))
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/config")
async def get_config():
    """Tell the UI whether apply (approve/autopilot) is available. In sandbox
    mode this reflects the tools pod's ALLOW_APPLY (where writes execute)."""
    return {"apply_enabled": toolclient.apply_enabled()}


@app.post("/apply")
async def apply_endpoint(req: ApplyRequest):
    """Execute a whitelisted remediation on a named target, then verify it.

    The same bounded path autopilot uses, gated by ALLOW_APPLY + the whitelist +
    a valid target name, triggered by the user's explicit approval instead.
    """
    async def event_stream():
        if not config.ALLOW_APPLY:
            yield sse("response", text="Apply is disabled on this server.")
            yield "data: [DONE]\n\n"
            return
        entry = remediation.REMEDIATIONS.get(req.action)
        if not entry or not extract.valid_k8s_name(req.target.split("/")[-1]):
            yield sse("response", text="Unknown or invalid remediation request.")
            yield "data: [DONE]\n\n"
            return
        loop = asyncio.get_event_loop()
        yield sse("step_start", name="Apply fix", note=f"{req.action} on {req.target}")
        t = time.perf_counter()
        result = await loop.run_in_executor(None, toolclient.apply, req.action, req.target)
        yield sse("step_done", name="Apply fix", ms=round((time.perf_counter() - t) * 1000), detail=result["message"])
        if result["ok"]:
            yield sse("step_start", name="Verify (CPU)", note=f"re-checking {req.target}")
            t = time.perf_counter()
            check = await loop.run_in_executor(None, toolclient.verify, req.action, req.target)
            yield sse("step_done", name="Verify (CPU)", ms=round((time.perf_counter() - t) * 1000), detail=check["message"])
            status = "Applied and verified" if check["ok"] else "Applied, but verification failed"
            yield sse("response", text=f"**{status}.** {check['message']}")
        else:
            yield sse("response", text=f"**Could not apply:** {result['message']}")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)
