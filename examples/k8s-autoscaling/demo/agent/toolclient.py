"""Client for the cluster-tools execution layer.

Single seam between the orchestrator and the code that touches the cluster. If
TOOLSVC_URL is set, every call is forwarded to the sandbox pod over HTTP (the
orchestrator then holds no cluster RBAC). If it is empty, the same calls run
in-process against the local functions, preserving single-pod dev mode.

All functions are synchronous; the graph already invokes them via
run_in_executor so the event loop is never blocked.
"""
import httpx

from . import config, remediation, tools

_TIMEOUT = 30.0


def _post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.post(f"{config.TOOLSVC_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()


def available() -> bool:
    """Whether cluster tools can run. Sandboxed: trust the operator's
    TOOLS_ENABLED (the sandbox owns the credentials and degrades gracefully if
    unreachable). Local: reflect whether in-process k8s creds loaded."""
    if config.TOOLSVC_URL:
        return config.TOOLS_ENABLED
    return tools.available()


def apply_enabled() -> bool:
    """Whether write/apply is on. Sandboxed: the tools pod owns ALLOW_APPLY and
    reports it via /health. Local: the orchestrator's own ALLOW_APPLY."""
    if not config.TOOLSVC_URL:
        return config.ALLOW_APPLY
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{config.TOOLSVC_URL}/health")
            r.raise_for_status()
            return bool(r.json().get("apply"))
    except Exception:
        return False
    """Whether cluster tools can run. Sandboxed: trust the operator's
    TOOLS_ENABLED (the sandbox owns the credentials and degrades gracefully if
    unreachable). Local: reflect whether in-process k8s creds loaded."""
    if config.TOOLSVC_URL:
        return config.TOOLS_ENABLED
    return tools.available()


def run_tool(tool: str, args: dict) -> str:
    if not config.TOOLSVC_URL:
        return tools.run_tool(tool, args)
    try:
        return _post("/run_tool", {"tool": tool, "args": args}).get("output", "")
    except Exception as e:
        return f"Tools service unavailable: {e}"


def detect_remediation(query: str) -> dict | None:
    if not config.TOOLSVC_URL:
        return remediation.detect_remediation(query)
    try:
        return _post("/detect_remediation", {"query": query}).get("remediation")
    except Exception:
        # Remediation is best-effort; a sandbox hiccup must not break answering.
        return None


def apply(action: str, target: str) -> dict:
    if not config.TOOLSVC_URL:
        entry = remediation.REMEDIATIONS.get(action)
        return entry[0](target) if entry else {"ok": False, "message": f"Unknown action: {action}"}
    try:
        return _post("/apply", {"action": action, "target": target})
    except Exception as e:
        return {"ok": False, "message": f"Tools service unavailable: {e}"}


def verify(action: str, target: str) -> dict:
    if not config.TOOLSVC_URL:
        entry = remediation.REMEDIATIONS.get(action)
        return entry[1](target) if entry else {"ok": False, "message": f"Unknown action: {action}"}
    try:
        return _post("/verify", {"action": action, "target": target})
    except Exception as e:
        return {"ok": False, "message": f"Tools service unavailable: {e}"}
