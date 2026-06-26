"""Sandboxed cluster-tools service (runs in its own pod).

This is the ONLY workload that holds Kubernetes credentials. The orchestrator
decides which tool to run and with what arguments (cheap, credential-free
parsing) and calls this service over HTTP; the actual cluster reads (and the
gated, whitelisted writes) happen here, isolated from the UI/LLM front door.

Run with:  uvicorn agent.toolsvc:app --host 0.0.0.0 --port 8080

Endpoints (all JSON):
  GET  /health
  POST /run_tool            {tool, args}        -> {output}
  POST /detect_remediation  {query}             -> {remediation: {...}|null}
  POST /apply               {action, target}    -> {ok, message}
  POST /verify              {action, target}    -> {ok, message}

There is no auth: restrict ingress to the orchestrator with a NetworkPolicy and
keep it behind a ClusterIP (never expose it outside the cluster).
"""
from fastapi import FastAPI
from pydantic import BaseModel

from . import extract, remediation, tools
from . import config

app = FastAPI()


class RunToolRequest(BaseModel):
    tool: str
    args: dict = {}


class QueryRequest(BaseModel):
    query: str


class RemediationRequest(BaseModel):
    action: str
    target: str


@app.on_event("startup")
async def _startup():
    tools.init_k8s()


@app.get("/health")
async def health():
    return {"status": "ok", "k8s": tools.available(), "apply": config.ALLOW_APPLY}


@app.post("/run_tool")
async def run_tool(req: RunToolRequest):
    return {"output": tools.run_tool(req.tool, req.args)}


@app.post("/detect_remediation")
async def detect_remediation(req: QueryRequest):
    return {"remediation": remediation.detect_remediation(req.query)}


def _resolve(action: str):
    """Look up a whitelisted remediation, rejecting unknown actions and any
    target name that fails k8s-name validation upstream."""
    return remediation.REMEDIATIONS.get(action)


@app.post("/apply")
async def apply(req: RemediationRequest):
    entry = _resolve(req.action)
    if not entry or not extract.valid_k8s_name(req.target.split("/")[-1]):
        return {"ok": False, "message": "Unknown or invalid remediation request."}
    return entry[0](req.target)


@app.post("/verify")
async def verify(req: RemediationRequest):
    entry = _resolve(req.action)
    if not entry or not extract.valid_k8s_name(req.target.split("/")[-1]):
        return {"ok": False, "message": "Unknown or invalid remediation request."}
    return entry[1](req.target)
