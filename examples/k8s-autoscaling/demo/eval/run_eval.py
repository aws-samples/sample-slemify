#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""End-to-end accuracy scorecard for the k8s-autoscaling agent.

Runs every case in cases.yaml through the live orchestrator's /query endpoint
(real triage -> retrieval -> auditor) and scores five things per case:

  1. triage      — did triage land the right category / reject noise
  2. must_not_say — deterministic: did the answer state a known-wrong claim
  3. correctness  — LLM-as-judge (Bedrock) vs the authoritative points
  4. calibration  — abstained when it should; didn't invent a problem on a valid
                    config; resisted a false premise
  5. hallucination — judge flag for fabricated API identifiers/behaviors

It writes a timestamped scorecard to eval/results/ and, if a baseline exists,
prints per-case regressions/improvements so a change that fixes one case and
breaks another is caught immediately.

Usage:
  # port-forward the orchestrator first:
  #   kubectl port-forward -n slemify svc/k8s-autoscaling-orchestrator 8000:80
  python3 run_eval.py                      # run all cases, print scorecard
  python3 run_eval.py --save-baseline      # also store as the comparison baseline
  python3 run_eval.py --only drift,minvalues-valid   # run a subset by id

Env:
  ORCHESTRATOR_URL  (default http://localhost:8000)
  JUDGE_MODEL       (default eu.anthropic.claude-sonnet-4-5-20250929-v1:0)
  AWS_REGION        (judge runs on Bedrock; uses the default AWS credentials)
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import boto3
import httpx
import yaml

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

_bedrock = boto3.client("bedrock-runtime")


# --- Orchestrator I/O ---

def query_agent(text: str) -> dict:
    """POST to /query and collect the final answer, triage detail, and model.

    Resets the answer buffer on answer_reset so only the final (post-retry)
    answer is scored.
    """
    triage, model, buf = "", "", []
    with httpx.Client(timeout=300) as c:
        with c.stream("POST", f"{ORCHESTRATOR_URL}/query",
                      json={"text": text, "autopilot": False}) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                t = ev.get("type")
                if t == "token":
                    buf.append(ev.get("text", ""))
                elif t == "response":
                    buf.append(ev.get("text", ""))
                elif t == "model":
                    model = ev.get("name", "")
                elif t == "answer_reset":
                    buf = []
                elif t == "step_done" and "Triage" in ev.get("name", ""):
                    triage = ev.get("detail", "")
    return {"triage": triage, "model": model, "answer": "".join(buf).strip()}


# --- Judge ---

JUDGE_PROMPT = """You are grading a Kubernetes-autoscaling assistant's answer for factual correctness. Be strict and base your grade on Karpenter v1 / KEDA / HPA v2 reality.

USER QUESTION:
{query}

ASSISTANT ANSWER:
{answer}

AUTHORITATIVE POINTS the answer should make (judge semantically, not verbatim):
{must_include}

CONTEXT FLAGS:
- The pasted config is actually VALID: {config_is_valid}
- The question's premise is FALSE and must be corrected, not agreed with: {leading_premise}
- The docs do not cover this; the assistant SHOULD abstain / say it cannot confirm: {should_abstain}

Return ONLY a JSON object, no prose:
{{"verdict": "pass|partial|fail",
  "abstained": true|false,
  "invented_problem": true|false,
  "followed_wrong_premise": true|false,
  "hallucinated": true|false,
  "reason": "one short sentence"}}

Where:
- verdict "pass" = factually correct AND makes the authoritative points; "partial" = mostly right but missing or hedged; "fail" = wrong, misleading, or misses the point.
- invented_problem = claimed something is wrong/needs fixing in a config that is actually valid.
- followed_wrong_premise = accepted a false premise instead of correcting it.
- hallucinated = stated an API field/value/behavior that does not exist or is incorrect."""


def judge(case: dict, answer: str) -> dict:
    must = case.get("must_include") or []
    prompt = JUDGE_PROMPT.format(
        query=case["query"].strip(),
        answer=answer or "(empty)",
        must_include="\n".join(f"- {m}" for m in must) or "(none)",
        config_is_valid=case.get("config_is_valid", False),
        leading_premise=case.get("leading_premise", False),
        should_abstain=case.get("should_abstain", False),
    )
    resp = _bedrock.converse(
        modelId=JUDGE_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 500, "temperature": 0},
    )
    text = resp["output"]["message"]["content"][0]["text"]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"verdict": "fail", "reason": f"unparseable judge output: {text[:120]}"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"verdict": "fail", "reason": f"bad judge json: {text[:120]}"}


# --- Scoring ---

def score_case(case: dict, result: dict) -> dict:
    answer = result["answer"]
    triage = result["triage"].lower()
    checks = {}

    # 1. triage
    if case.get("should_reject"):
        checks["triage"] = "reject" in triage or "does not look like" in answer.lower()
    else:
        exp = case.get("expected_category", "").lower()
        checks["triage"] = (exp in triage) if exp else True

    # 2. must_not_say (deterministic, hard fail)
    violations = [p for p in (case.get("must_not_say") or [])
                  if p.lower() in answer.lower()]
    checks["must_not_say"] = not violations

    # Reject cases: triage is the whole story.
    if case.get("should_reject"):
        status = "pass" if checks["triage"] else "fail"
        return {"status": status, "checks": checks, "violations": violations,
                "judge": {}, "answer_chars": len(answer)}

    # 3-5. judge-based correctness + calibration + hallucination
    j = judge(case, answer)
    verdict = j.get("verdict", "fail")
    calibration_fail = (
        (case.get("should_abstain") and not j.get("abstained")) or
        (case.get("config_is_valid") and j.get("invented_problem")) or
        (case.get("leading_premise") and j.get("followed_wrong_premise")))
    checks["calibration"] = not calibration_fail
    checks["correctness"] = verdict == "pass"
    checks["not_hallucinated"] = not j.get("hallucinated", False)

    if violations or calibration_fail or verdict == "fail":
        status = "fail"
    elif verdict == "partial" or j.get("hallucinated"):
        status = "partial"
    else:
        status = "pass"
    return {"status": status, "checks": checks, "violations": violations,
            "judge": j, "answer_chars": len(answer)}


# --- Runner ---

def _aggregate(statuses: list) -> str:
    """Collapse repeated-run statuses into one, breaking ties toward the worse
    outcome so flaky cases are not reported as clean."""
    order = {"error": 0, "fail": 1, "partial": 2, "pass": 3}
    counts = {}
    for s in statuses:
        counts[s] = counts.get(s, 0) + 1
    best = max(counts.values())
    tied = [s for s, c in counts.items() if c == best]
    return min(tied, key=lambda s: order.get(s, 0))


def run_case(case: dict, repeat: int) -> dict:
    """Run a case `repeat` times; return aggregated status + per-run detail."""
    runs, judges, secs = [], [], []
    last = {}
    for _ in range(repeat):
        t0 = time.perf_counter()
        try:
            result = query_agent(case["query"])
            sc = score_case(case, result)
        except Exception as e:
            sc = {"status": "error", "checks": {}, "violations": [],
                  "judge": {"reason": str(e)[:160]}}
        secs.append(time.perf_counter() - t0)
        runs.append(sc["status"])
        judges.append(sc.get("judge", {}))
        last = sc
    status = _aggregate(runs)
    passes = sum(1 for s in runs if s == "pass")
    return {"id": case["id"], "topic": case.get("topic", ""),
            "status": status, "runs": runs, "pass_rate": f"{passes}/{repeat}",
            "violations": last.get("violations", []),
            "judge": judges[-1], "seconds": round(sum(secs), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(HERE, "cases.yaml"))
    ap.add_argument("--only", default="", help="comma-separated case ids to run")
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per case; status is the majority (ties -> worse). "
                         "Use >1 to average out the auditor's run-to-run variance.")
    ap.add_argument("--save-baseline", action="store_true")
    args = ap.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        cases = yaml.safe_load(f)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    print(f"Running {len(cases)} cases against {ORCHESTRATOR_URL} "
          f"(repeat={args.repeat})\n")
    for case in cases:
        row = run_case(case, args.repeat)
        icon = {"pass": "PASS", "partial": "PART", "fail": "FAIL",
                "error": "ERR "}.get(row["status"], "?")
        reason = row.get("judge", {}).get("reason", "")
        viol = f" | said: {row['violations']}" if row.get("violations") else ""
        rate = f"({row['pass_rate']})" if args.repeat > 1 else ""
        print(f"  [{icon}] {row['id']:<32} {row['seconds']:5.1f}s {rate}  {reason}{viol}")
        rows.append(row)

    # Aggregate
    counts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    n = len(rows)
    print(f"\n=== Scorecard: {counts['pass']}/{n} pass, "
          f"{counts['partial']} partial, {counts['fail']} fail, "
          f"{counts['error']} error ===")

    scorecard = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "orchestrator": ORCHESTRATOR_URL, "counts": counts,
                 "n": n, "rows": rows}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(RESULTS_DIR, f"scorecard-{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"Wrote {out}")

    # Diff vs baseline
    baseline_path = os.path.join(RESULTS_DIR, "baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            base = {r["id"]: r["status"] for r in json.load(f)["rows"]}
        order = {"fail": 0, "error": 0, "partial": 1, "pass": 2}
        regressions, improvements = [], []
        for r in rows:
            b = base.get(r["id"])
            if b is None:
                continue
            if order.get(r["status"], 0) < order.get(b, 0):
                regressions.append(f"{r['id']}: {b} -> {r['status']}")
            elif order.get(r["status"], 0) > order.get(b, 0):
                improvements.append(f"{r['id']}: {b} -> {r['status']}")
        print("\n--- vs baseline ---")
        print("  regressions:  " + (", ".join(regressions) or "none"))
        print("  improvements: " + (", ".join(improvements) or "none"))

    if args.save_baseline:
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(scorecard, f, indent=2)
        print(f"Saved baseline -> {baseline_path}")

    # Non-zero exit if anything failed (useful for CI/hooks).
    sys.exit(1 if (counts["fail"] or counts["error"]) else 0)


if __name__ == "__main__":
    main()
