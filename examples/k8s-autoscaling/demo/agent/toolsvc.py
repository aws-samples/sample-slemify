"""Sandboxed cluster-tools service (runs in its own pod).

This is the ONLY workload that holds Kubernetes credentials. The orchestrator
decides which tool to run and with what arguments (cheap, credential-free
parsing) and calls this service over HTTP; the actual cluster reads (and the
gated, whitelisted writes) happen here, isolated from the UI/LLM front door.

Run with:  uvicorn agent.toolsvc:app --host 0.0.0.0 --port 8080

Endpoints (all JSON):
  GET  /health
  POST /run_tool            {tool, args}                  -> {output}
  POST /detect_remediation  {query}                       -> {remediation: {...}|null}
  POST /named_target        {query}                       -> {target: {kind, target}|null}
  POST /plan_fix            {kind, target, field, value}  -> {ok, message, ...}
  POST /apply_fix           {kind, target, field, value}  -> {ok, message}
  POST /verify_fix          {kind, target, field, value}  -> {ok, message}

Every write endpoint re-validates (kind, field, value) against
patch_schema.PATCH_SCHEMA itself (remediation.plan_patch/apply_patch do this) —
this service never trusts the orchestrator's word that a proposal is valid,
even though the orchestrator already checked it once. See PERMISSIONS.md for
the human-readable authority boundary this schema (plus RBAC) enforces.

There is no auth: restrict ingress to the orchestrator with a NetworkPolicy and
keep it behind a ClusterIP (never expose it outside the cluster).
"""
from fastapi import FastAPI
from pydantic import BaseModel

from . import remediation, tools
from . import config

app = FastAPI()


class RunToolRequest(BaseModel):
    tool: str
    args: dict = {}


class QueryRequest(BaseModel):
    query: str


class FixRequest(BaseModel):
    kind: str
    target: str
    field: str
    value: str = ""


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


@app.post("/named_target")
async def named_target(req: QueryRequest):
    return {"target": remediation.named_target(req.query)}


@app.post("/plan_fix")
async def plan_fix(req: FixRequest):
    return remediation.plan_patch(req.kind, req.target, req.field, req.value)


@app.post("/apply_fix")
async def apply_fix(req: FixRequest):
    return remediation.apply_patch(req.kind, req.target, req.field, req.value)


@app.post("/verify_fix")
async def verify_fix(req: FixRequest):
    return remediation.verify_patch(req.kind, req.target, req.field, req.value)
