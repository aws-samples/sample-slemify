"""Metrics: the gate pass rate, and the per-query scoreboard.

Two things are measured here.

1. Gate outcomes: how often the CPU SLM's draft clears the faithfulness gate on
   the first try. The cost model of this architecture hinges on that number:
   below roughly 87-92%, the escalation path (gate -> escalate -> gate again)
   makes CPU-first more expensive per query than calling the LLM directly.

2. The scoreboard: for every query, the Bedrock tokens spent (by model and by
   purpose), the resulting USD, and the wall-clock of each step. Aggregated per
   seat configuration (config.seats()), so moving one seat and re-running the
   same queries shows what that move changed in cost and latency. Quality
   comes from the eval, not from here.

Per-query accounting uses a context variable. server.py opens a meter before
running the graph; every Bedrock call site charges it; graph nodes run their
blocking calls with asyncio.to_thread, which copies the context into the
worker thread so the charge lands on the right query.

Two sinks, both cheap: JSON lines on stdout (kubectl logs is the durable
record) and in-memory aggregates at GET /stats (reset on restart).
"""
import contextvars
import json
import statistics
import threading
import time
from collections import Counter, defaultdict

from . import config

_lock = threading.Lock()
_started = time.time()
_counters: Counter = Counter()
_by_category: dict[str, Counter] = {}

# Per-configuration samples for the scoreboard. Bounded so a long-running pod
# does not grow without limit; the JSON lines hold the full history.
_MAX_SAMPLES = 2000
_queries: dict[str, list[dict]] = defaultdict(list)


def _emit(event: str, **fields):
    """One JSON line per event on stdout, greppable via kubectl logs."""
    print(json.dumps({"metric": event, "ts": round(time.time(), 3), **fields}),
          flush=True)


# --- Gate outcomes ---

def record_gate_check(category: str, used_llm: bool, escalate: bool,
                      attempts: int, reason: str = ""):
    """Every gate invocation, including re-checks of escalated answers."""
    source = "llm" if used_llm else "slm"
    verdict = "flagged" if escalate else "supported"
    with _lock:
        _counters[f"gate_{source}_{verdict}"] += 1
        # First-attempt SLM checks are the pass-rate denominator: attempts==1
        # means this is the SLM's first draft for the query (no refine loop yet).
        if source == "slm" and attempts == 1:
            _counters["slm_first_checks"] += 1
            if not escalate:
                _counters["slm_first_pass"] += 1
    _emit("gate_check", category=category, source=source, verdict=verdict,
          attempts=attempts, reason=reason[:120])


def record_outcome(category: str, outcome: str):
    """Terminal resolution: slm_pass | llm_pass | abstain | propose."""
    with _lock:
        _counters[f"outcome_{outcome}"] += 1
        _by_category.setdefault(category or "unknown", Counter())[outcome] += 1
    _emit("outcome", category=category, outcome=outcome)


# --- Per-query meter ---

class Meter:
    """Everything one query spent: Bedrock calls (tokens and USD) and step
    timings. Mutated in place from whichever thread the call runs on."""

    def __init__(self):
        self.calls: list[dict] = []
        self.steps: dict[str, int] = {}
        self._lock = threading.Lock()

    def charge(self, model_id: str, purpose: str, tokens_in: int, tokens_out: int):
        p_in, p_out = config.price_per_million(model_id)
        usd = (tokens_in * p_in + tokens_out * p_out) / 1_000_000
        with self._lock:
            self.calls.append({"model": model_id, "purpose": purpose,
                               "tokens_in": tokens_in, "tokens_out": tokens_out,
                               "usd": round(usd, 6)})

    def step(self, name: str, ms: int):
        with self._lock:
            # A step that runs twice (refine loop) accumulates.
            self.steps[name] = self.steps.get(name, 0) + ms

    def summary(self) -> dict:
        with self._lock:
            return {
                "usd": round(sum(c["usd"] for c in self.calls), 6),
                "bedrock_calls": len(self.calls),
                "tokens_in": sum(c["tokens_in"] for c in self.calls),
                "tokens_out": sum(c["tokens_out"] for c in self.calls),
                "by_purpose": {
                    p: round(sum(c["usd"] for c in self.calls if c["purpose"] == p), 6)
                    for p in sorted({c["purpose"] for c in self.calls})},
                "steps_ms": dict(self.steps),
            }


_current: contextvars.ContextVar[Meter | None] = contextvars.ContextVar("meter", default=None)


def open_meter() -> Meter:
    m = Meter()
    _current.set(m)
    return m


def charge(model_id: str, purpose: str, tokens_in: int, tokens_out: int):
    """Record a Bedrock call against the current query. No-op outside a query
    (warmup, scripts), so call sites never need to check."""
    m = _current.get()
    if m is not None:
        m.charge(model_id, purpose, tokens_in, tokens_out)


def usage_from_converse(resp: dict) -> tuple[int, int]:
    u = resp.get("usage", {}) or {}
    return int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0))


def seats_key(seats: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(seats.items()))


def record_query(seats: dict, meter: Meter, total_ms: int, escalated: bool):
    """Close the books on one query: emit the line and keep the sample for the
    per-configuration aggregate."""
    s = meter.summary()
    sample = {"usd": s["usd"], "total_ms": total_ms, "steps_ms": s["steps_ms"],
              "escalated": escalated, "tokens_in": s["tokens_in"],
              "tokens_out": s["tokens_out"]}
    key = seats_key(seats)
    with _lock:
        q = _queries[key]
        q.append(sample)
        if len(q) > _MAX_SAMPLES:
            del q[: len(q) - _MAX_SAMPLES]
    _emit("query", seats=seats, total_ms=total_ms, escalated=escalated, **s)


def _p(values: list, q: float):
    if not values:
        return None
    vs = sorted(values)
    idx = min(len(vs) - 1, max(0, round(q * (len(vs) - 1))))
    return vs[idx]


def scoreboard() -> dict:
    """Per seat configuration: how many queries, what they cost, how long they
    took, and the per-step p50. This is the row attendees copy onto the
    worksheet after each seat move."""
    with _lock:
        snap = {k: list(v) for k, v in _queries.items()}
    out = {}
    for key, samples in snap.items():
        usd = [s["usd"] for s in samples]
        ms = [s["total_ms"] for s in samples]
        step_names = sorted({n for s in samples for n in s["steps_ms"]})
        steps = {n: _p([s["steps_ms"][n] for s in samples if n in s["steps_ms"]], 0.5)
                 for n in step_names}
        out[key] = {
            "queries": len(samples),
            "bedrock_usd_per_query_mean": round(statistics.fmean(usd), 6) if usd else None,
            "bedrock_usd_per_query_p50": _p(usd, 0.5),
            "total_ms_p50": _p(ms, 0.5),
            "total_ms_p95": _p(ms, 0.95),
            "escalation_rate": round(sum(1 for s in samples if s["escalated"]) / len(samples), 4),
            "steps_ms_p50": steps,
        }
    return out


def snapshot() -> dict:
    """Live counters, the headline rates, and the scoreboard, for GET /stats."""
    with _lock:
        c = dict(_counters)
        cats = {k: dict(v) for k, v in _by_category.items()}
    checks = c.get("slm_first_checks", 0)
    passes = c.get("slm_first_pass", 0)
    outcomes = {k[len("outcome_"):]: v for k, v in c.items()
                if k.startswith("outcome_")}
    total_outcomes = sum(outcomes.values())
    return {
        "uptime_seconds": round(time.time() - _started),
        # The headline number: of the SLM's first drafts, how many cleared the
        # gate without escalation. Compare against the ~87-92% breakeven.
        "slm_first_pass_rate": round(passes / checks, 4) if checks else None,
        "slm_first_checks": checks,
        "slm_first_pass": passes,
        "outcomes": outcomes,
        "outcomes_total": total_outcomes,
        "by_category": cats,
        "counters": c,
        "scoreboard": scoreboard(),
        "cpu_pool_usd_per_hour": config.CPU_POOL_USD_PER_HOUR or None,
        "note": ("counters reset on pod restart; the JSON-line events in pod "
                 "logs are the durable record. bedrock_usd counts Bedrock tokens "
                 "only; CPU pods are hourly capacity, see cpu_pool_usd_per_hour"),
    }
