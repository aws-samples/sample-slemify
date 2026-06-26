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
from itertools import zip_longest

import boto3
import httpx
import yaml

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
# The judge grounds its grading in the same knowledge base the agent uses, so it
# verifies the answer's claims against the authoritative docs instead of its own
# (possibly stale) memory. Retrieved broadly on the question AND the answer's own
# wording, so claim-specific facts (e.g. a policy the answer names) are surfaced
# even when the question doesn't mention them.
EMBED_URL = os.environ.get("EMBEDDING_URL", "http://localhost:8083")
KNOWLEDGE_URL = os.environ.get("KNOWLEDGE_URL", "http://localhost:9200/k8s-autoscaling-knowledge")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

_bedrock = boto3.client("bedrock-runtime")


def fetch_reference(query: str, answer: str) -> str:
    """Pull authoritative docs from the knowledge base to ground the judge.

    Unions dense (semantic) hits on the question with lexical hits on both the
    question and the answer, so a fact the answer asserts (a field/policy name)
    is retrievable for verification even if the question never mentioned it.
    Returns "" if the retrieval endpoints are unavailable (judge degrades to
    grading on the authoritative points alone)."""
    def _hits(body):
        r = httpx.post(f"{KNOWLEDGE_URL}/_search", json=body, timeout=30)
        r.raise_for_status()
        return [h["_source"]["text"] for h in r.json()["hits"]["hits"]]

    try:
        vec_q = httpx.post(f"{EMBED_URL}/embed", json={"inputs": query[:8000]}, timeout=30).json()[0]
        dense_q = _hits({"size": 15, "query": {"knn": {"embedding": {"vector": vec_q, "k": 15}}},
                        "_source": ["text"]})
        lex_q = _hits({"size": 10, "query": {"match": {"text": query[:1000]}}, "_source": ["text"]})
        lex_a = _hits({"size": 15, "query": {"match": {"text": answer[:2000]}}, "_source": ["text"]}) if answer else []
        # Dense search on the ANSWER too: a real identifier the answer states in a
        # different surface form than the corpus (e.g. MIN_VALUES_POLICY vs the
        # docs' MinValuesPolicy / --min-values-policy) won't match by BM25 token,
        # but its semantics still retrieve the proving chunk. Without this the
        # judge can falsely flag a true claim as fabricated.
        dense_a = []
        if answer:
            vec_a = httpx.post(f"{EMBED_URL}/embed", json={"inputs": answer[:8000]}, timeout=30).json()[0]
            dense_a = _hits({"size": 15, "query": {"knn": {"embedding": {"vector": vec_a, "k": 15}}},
                            "_source": ["text"]})
        # Round-robin across the four sources so answer-grounding chunks always get
        # slots (a question-dense flood can't crowd them out before the cap).
        seen, chunks = set(), []
        for group in zip_longest(dense_q, dense_a, lex_a, lex_q):
            for c in group:
                if c and c not in seen:
                    seen.add(c)
                    chunks.append(c)
        return "\n\n---\n\n".join(chunks[:18])
    except Exception as e:
        print(f"  (reference lookup unavailable: {e}; judging on authoritative points only)")
        return ""


# --- Orchestrator I/O ---

def query_agent(text: str, attempts: int = 3) -> dict:
    """Query the orchestrator, retrying transient transport errors (e.g. a
    dropped port-forward) so infra blips are not scored as model failures."""
    last_exc = None
    for i in range(attempts):
        try:
            return _query_once(text)
        except httpx.TransportError as e:  # ConnectError, ReadError, RemoteProtocolError, ...
            last_exc = e
            if i < attempts - 1:
                time.sleep(3)
    raise last_exc


def _query_once(text: str) -> dict:
    """POST to /query and collect the final answer, triage detail, and model.

    Resets the answer buffer on answer_reset so only the final (post-retry)
    answer is scored.
    """
    triage, model, buf = "", "", []
    autopilot = os.environ.get("EVAL_AUTOPILOT", "false").lower() in ("1", "true", "yes")
    with httpx.Client(timeout=300) as c:
        with c.stream("POST", f"{ORCHESTRATOR_URL}/query",
                      json={"text": text, "autopilot": autopilot}) as r:
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

JUDGE_PROMPT = """You are grading a Kubernetes-autoscaling assistant's answer.

You are given two sources of truth, in priority order:
1. AUTHORITATIVE POINTS — verified facts the answer should make.
2. REFERENCE DOCUMENTATION — passages retrieved from the same knowledge base the
   assistant uses. Treat this as the source of record for whether a specific
   claim (a field, value, policy name, behavior) is real.

Grade against these two sources and the user question. Do NOT use your own
outside knowledge to overrule them. In particular, if the answer states a field
or behavior that appears in the REFERENCE DOCUMENTATION, it is real — do not call
it fabricated just because it is not in the authoritative points or because you
do not recall it.

USER QUESTION:
{query}

ASSISTANT ANSWER:
{answer}

AUTHORITATIVE POINTS (ground truth; judge semantically, not verbatim):
{must_include}

REFERENCE DOCUMENTATION (authoritative source of record):
{reference}

HOW TO APPLY THE CASE FLAGS (these describe the verified reality of this case):
- config_is_valid = {config_is_valid}
    If true: the pasted config is correct. An answer that invents a problem or
    tells the user to "fix" it is a FAIL (set invented_problem=true).
    If false: the config genuinely has the problem described by the authoritative
    points. An answer that calls it valid/fine is a FAIL.
- leading_premise = {leading_premise}
    If true: the question contains a FALSE premise. A correct answer corrects it
    and reaches the authoritative points above. Judge premise-handling BY THE
    AUTHORITATIVE POINTS, not by parsing the question's wording: if the answer
    asserts what the authoritative points say, it correctly rejected the false
    premise (this axis PASSES). Only set followed_wrong_premise=true if the answer
    asserts the OPPOSITE of the authoritative points (i.e. it agrees with the
    false premise). Phrases like "that premise is incorrect" followed by the
    authoritative points are a CORRECT rejection, not a failure.
- should_abstain = {should_abstain}
    If true: the docs do not cover this. The answer must decline or say it cannot
    confirm an exact answer. Answering with invented specifics is a FAIL
    (set abstained=false). Abstaining correctly is a PASS.

Return ONLY a JSON object, no prose:
{{"verdict": "pass|partial|fail",
  "abstained": true|false,
  "invented_problem": true|false,
  "followed_wrong_premise": true|false,
  "hallucinated": true|false,
  "reason": "one short sentence"}}

Where:
- verdict "pass" = makes the authoritative points and is consistent with them, the
  reference documentation, and the flags above. An answer that makes the points
  clearly but ALSO adds honest notes about what it could not verify (calibrated
  uncertainty) is still a PASS — do NOT downgrade for appropriate hedging or for
  saying "I couldn't confirm X" when the core points are stated. "partial" =
  hedges so much that it never commits to an authoritative point, or is missing a
  point; "fail" = contradicts an authoritative point or the reference, violates a
  flag rule above, or misses the point of the question.
- hallucinated = stated an API field/value/behavior that CONTRADICTS the reference
  documentation or authoritative points, or that appears in neither and is not a
  basic, well-established fact. Do NOT flag a claim that the reference supports."""


def judge(case: dict, answer: str) -> dict:
    must = case.get("must_include") or []
    reference = fetch_reference(case["query"], answer or "")
    prompt = JUDGE_PROMPT.format(
        query=case["query"].strip(),
        answer=answer or "(empty)",
        must_include="\n".join(f"- {m}" for m in must) or "(none)",
        reference=reference or "(reference lookup unavailable)",
        config_is_valid=case.get("config_is_valid", False),
        leading_premise=case.get("leading_premise", False),
        should_abstain=case.get("should_abstain", False),
    )
    resp = _bedrock.converse(
        modelId=JUDGE_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 800, "temperature": 0},
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
    last_answer, last_model = "", ""
    for _ in range(repeat):
        t0 = time.perf_counter()
        try:
            result = query_agent(case["query"])
            last_answer, last_model = result.get("answer", ""), result.get("model", "")
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
            "judge": judges[-1], "seconds": round(sum(secs), 1),
            "model": last_model, "answer": last_answer}


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
