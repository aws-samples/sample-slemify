#!/usr/bin/env python3
"""Held-out triage accuracy for whichever model runs the TRIAGE step.

Exact-match, deterministic, no judge: the one quality number for the router
step that cannot wobble between attendees. Runs the same `agent.classify`
code the orchestrator runs, so TRIAGE=llm scores the Bedrock router and
TRIAGE=classifier scores the ONNX classifier through the same parser.

Usage (port-forward the classifier first for TRIAGE=classifier):
  kubectl port-forward -n slemify svc/k8s-autoscaling-triage-inference 8081:8080
  TRIAGE=classifier python3 eval/triage_acc.py
  TRIAGE=llm        python3 eval/triage_acc.py

A throttled or failed call is retried (4 attempts, backoff 2, 4, 8 s). A
message that still fails is an [ERR ] row: it is not scored, so it does not
count as a miss, and the summary says how many messages were not scored.

Prints per-label accuracy, the confusion pairs, latency p50, and the cost: for
TRIAGE=llm the frontier model's cost per call, for TRIAGE=classifier the CPU
price per 1M classifications from the latest `make bench-triage` run. Writes
eval/results/triage-<step>-<stamp>.json. `make bench-triage` reads the cost
from the latest llm run to compare the two models on price.
"""
import argparse
import glob
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import httpx
import yaml
from botocore.exceptions import (ClientError, ConnectionClosedError,
                                 EndpointConnectionError, ReadTimeoutError)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # the demo dir, for `agent`
from agent import classify, config, metrics  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
ATTEMPTS = 4
BASE_DELAY_S = 2.0

# Bedrock error codes worth another try: throttling and transient service-side
# failures. Anything else (a bad request, missing model access) fails the
# message at once.
RETRYABLE_CODES = {
    "ThrottlingException", "ServiceUnavailableException", "ServiceUnavailable",
    "ModelNotReadyException", "InternalServerException", "ModelTimeoutException",
}


def load_set(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        items = yaml.safe_load(f)
    out = []
    for it in items:
        if "file" in it:
            with open(os.path.join(HERE, it["file"]), encoding="utf-8") as f:
                it["query"] = f.read()
        it.setdefault("accept", [it["label"]])
        if it["label"] not in it["accept"]:
            it["accept"].append(it["label"])
        out.append(it)
    return out


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code", "") in RETRYABLE_CODES
    # Transport failures: Bedrock endpoint (botocore) or the classifier forward (httpx).
    return isinstance(exc, (EndpointConnectionError, ReadTimeoutError,
                            ConnectionClosedError, httpx.TransportError))


def classify_with_retry(text: str, attempts: int = ATTEMPTS, base_delay: float = BASE_DELAY_S,
                        sleep=time.sleep) -> tuple[dict, float, int]:
    """Classify one message, retrying retryable errors with exponential backoff
    (base_delay x 2^n plus up to 0.5 s of jitter). Returns (prediction, usd,
    ms of the successful attempt). Each attempt opens its own meter, so a failed
    attempt adds no cost. Raises the last exception when no attempt succeeds."""
    for n in range(attempts):
        meter = metrics.open_meter()  # records the Bedrock cost when TRIAGE=llm
        t0 = time.perf_counter()
        try:
            pred = classify.classify(text)
        except Exception as e:
            if n == attempts - 1 or not is_retryable(e):
                raise
            sleep(base_delay * 2 ** n + random.uniform(0, 0.5))
            continue
        ms = round((time.perf_counter() - t0) * 1000)
        return pred, meter.summary()["usd"], ms
    raise RuntimeError("unreachable")


def error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:80]}"


def cpu_cost_line(results_dir: str) -> str:
    """The classifier's price per 1M classifications, from the newest
    `make bench-triage` result. A CPU price comes from CPU time and the node's
    hourly price, not from latency, so triage-acc does not compute one."""
    for path in sorted(glob.glob(os.path.join(results_dir, "bench-triage-*.json")), reverse=True):
        try:
            with open(path, encoding="utf-8") as f:
                bench = json.load(f)
        except (OSError, ValueError):
            continue
        if bench.get("usd_per_1m") is None:
            continue
        stamp = os.path.basename(path)[len("bench-triage-"):-len(".json")]
        instance = bench.get("instance_type") or "the triage node"
        return (f"  CPU cost: ${bench['usd_per_1m']:.2f} per 1M classifications on {instance} "
                f"(make bench-triage, {stamp})")
    return ("  CPU cost: no per-call bill; the classifier runs on a CPU node billed by the hour. "
            "make bench-triage prices it per 1M classifications.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(HERE, "triage-heldout.yaml"))
    ap.add_argument("--only-label", default="", help="restrict to one label")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args(argv)

    items = load_set(args.set)
    if args.only_label:
        items = [i for i in items if i["label"] == args.only_label]

    step = config.TRIAGE
    print(f"Triage step: {step}  ({len(items)} held-out queries)\n")

    per_label = defaultdict(lambda: [0, 0])  # label -> [correct, scored]
    confusions = Counter()
    latencies = []
    usd = []
    rows = []
    errors = 0
    for it in items:
        try:
            pred, cost, ms = classify_with_retry(it["query"])
        except Exception as e:
            errors += 1
            err = error_text(e)
            print(f"  [ERR ] {it['id']:<34} {it['label']:<18} -> error: {err}")
            rows.append({"id": it["id"], "label": it["label"], "accept": it["accept"],
                         "predicted": None, "confidence": None, "ok": None, "ms": None,
                         "error": err})
            continue
        latencies.append(ms)
        usd.append(cost)
        ok = pred["category"] in it["accept"]
        per_label[it["label"]][1] += 1
        if ok:
            per_label[it["label"]][0] += 1
        else:
            confusions[(it["label"], pred["category"])] += 1
        mark = "ok  " if ok else "MISS"
        print(f"  [{mark}] {it['id']:<34} {it['label']:<18} -> {pred['category']:<18} "
              f"{pred['confidence']:<7} {ms:>5} ms")
        rows.append({"id": it["id"], "label": it["label"], "accept": it["accept"],
                     "predicted": pred["category"], "confidence": pred["confidence"],
                     "ok": ok, "ms": ms})

    scored = sum(v[1] for v in per_label.values())
    correct = sum(v[0] for v in per_label.values())
    accuracy = correct / scored if scored else 0.0
    p50 = statistics.median(latencies) if latencies else 0
    not_scored = f"  ({errors} not scored: errors)" if errors else ""
    print(f"\n=== Triage accuracy [{step}]: {correct}/{scored} = {accuracy:.1%}  "
          f"latency p50 {p50:.0f} ms{not_scored} ===")
    if errors:
        print(f"  {errors} message(s) failed after {ATTEMPTS} attempts and were not scored. "
              f"Run make triage-acc again to score all {len(items)}.")
    usd_mean = statistics.fmean(usd) if usd else 0.0
    if usd_mean:
        print(f"  frontier cost: ${usd_mean:.6f} per call = ${usd_mean * 1e6:,.0f} per 1M calls")
    elif step == "classifier":
        print(cpu_cost_line(args.results_dir))
    for label in sorted(per_label):
        c, n = per_label[label]
        print(f"  {label:<18} {c}/{n}")
    if confusions:
        print("  confusions (expected -> predicted):")
        for (exp, got), n in confusions.most_common():
            print(f"    {exp} -> {got}: {n}")

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(args.results_dir, f"triage-{step}-{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        # accuracy is correct / scored; total is every message, scored or not.
        json.dump({"step": step, "accuracy": accuracy, "correct": correct,
                   "scored": scored, "errors": errors, "total": len(items),
                   "latency_p50_ms": p50,
                   "usd_per_call_mean": round(usd_mean, 8),
                   "per_label": {k: {"correct": v[0], "total": v[1]} for k, v in per_label.items()},
                   "rows": rows}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
