"""Gate-outcome metrics: the measured production pass rate.

The cost model of this architecture hinges on one number: how often the CPU
SLM's draft clears the faithfulness gate on the first try. Below ~87-92%,
the escalation path (gate -> escalate -> gate again) makes CPU-first MORE
expensive per query than calling the LLM directly. Until now that number was
only ever a scorecard proxy from a noisy eval; this module measures the real
thing in production.

Two sinks, both cheap:
  - JSON lines on stdout (one per event), so `kubectl logs` is a queryable
    record that survives pod restarts as long as logs are retained.
  - In-memory counters exposed at GET /stats for a live view.

Recorded events:
  - gate_check: every time the faithfulness gate runs (draft source, verdict).
  - outcome:    the terminal resolution of a query that reached the gate:
      slm_pass  — SLM draft accepted (the cheap path; the number to maximize)
      llm_pass  — SLM failed, escalated LLM answer accepted
      abstain   — even the escalated answer failed; calibrated fallback
      propose   — supervised mode deferred to the user for live evidence
                  (unresolved this turn; SLM draft did not clear the gate)
"""
import json
import threading
import time
from collections import Counter

_lock = threading.Lock()
_started = time.time()
_counters: Counter = Counter()
_by_category: dict[str, Counter] = {}


def _emit(event: str, **fields):
    """One JSON line per event on stdout — greppable via kubectl logs."""
    print(json.dumps({"metric": event, "ts": round(time.time(), 3), **fields}),
          flush=True)


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


def snapshot() -> dict:
    """Live counters + the headline rates, for GET /stats."""
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
        "note": ("counters reset on pod restart; the JSON-line events in pod "
                 "logs are the durable record"),
    }
