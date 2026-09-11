#!/usr/bin/env python3
"""Held-out triage accuracy for whoever holds the TRIAGE seat.

Exact-match, deterministic, no judge: the one quality number for the router
seat that cannot wobble between attendees. Runs the same `agent.classify`
code the orchestrator runs, so TRIAGE=llm scores the Bedrock router and
TRIAGE=classifier scores the ONNX classifier through the same parser.

Usage (port-forward the classifier first for TRIAGE=classifier):
  kubectl port-forward -n slemify svc/k8s-autoscaling-triage-inference 8081:8080
  TRIAGE=classifier python3 eval/triage_acc.py
  TRIAGE=llm        python3 eval/triage_acc.py

Prints per-label accuracy, the confusion pairs, and latency p50, and writes
eval/results/triage-<seat>-<stamp>.json.
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # the demo dir, for `agent`

from agent import classify, config  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(HERE, "triage-heldout.yaml"))
    ap.add_argument("--only-label", default="", help="restrict to one label")
    args = ap.parse_args()

    items = load_set(args.set)
    if args.only_label:
        items = [i for i in items if i["label"] == args.only_label]
    seat = config.TRIAGE
    print(f"Triage seat: {seat}  ({len(items)} held-out queries)\n")

    per_label = defaultdict(lambda: [0, 0])  # label -> [correct, total]
    confusions = Counter()
    latencies = []
    rows = []
    for it in items:
        t0 = time.perf_counter()
        try:
            pred = classify.classify(it["query"])
        except Exception as e:
            pred = {"category": f"error: {e}", "confidence": "unknown"}
        ms = round((time.perf_counter() - t0) * 1000)
        latencies.append(ms)
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

    total = sum(v[1] for v in per_label.values())
    correct = sum(v[0] for v in per_label.values())
    print(f"\n=== Triage accuracy [{seat}]: {correct}/{total} = {correct / total:.1%}  "
          f"latency p50 {statistics.median(latencies):.0f} ms ===")
    for label in sorted(per_label):
        c, n = per_label[label]
        print(f"  {label:<18} {c}/{n}")
    if confusions:
        print("  confusions (expected -> predicted):")
        for (exp, got), n in confusions.most_common():
            print(f"    {exp} -> {got}: {n}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(RESULTS_DIR, f"triage-{seat}-{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seat": seat, "accuracy": correct / total, "correct": correct,
                   "total": total, "latency_p50_ms": statistics.median(latencies),
                   "per_label": {k: {"correct": v[0], "total": v[1]} for k, v in per_label.items()},
                   "rows": rows}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
