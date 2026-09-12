# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Slemify report: what the served model does, measured, with a baseline.

Runs as a Kubernetes Job after the serving stage, next to the inference
Service. One report shape for every task family, organised the way a
placement decision is made:

  quality   the held-out numbers the training job wrote (metrics.json), next to
            a trivial baseline, split by where the held-out data came from
            (synthetic versus human-labeled), with the confusions and the
            reliability of the head's probabilities. For classification, an
            optional zero-shot frontier-model baseline on the same held-out set.
  latency   measured against the served endpoint, not the training job.
  serving   the node the pod landed on and its on-demand hourly rate, stated
            as what it is: fixed capacity, not a per-request price.

For generation (served stock, grounded by retrieval at query time) there is no
held-out label set, so the report is a serving profile: prefill and decode
rates, cold and warm time to first token, the model's bytes, and the
memory-bandwidth ceiling for the node. An optional grounded evaluation
(question, evidence, points the answer must make) is judged by a frontier
model when the expert config points at one.

The report never says "ready" or "not ready". It gives the numbers and, when
one of them is low, the order in which to look for the cause.

Environment (set by the Slemify CLI on the Job):
  BUCKET, PROJECT, TASK, INFERENCE_ENDPOINT, BEDROCK_MODEL, MAX_SAMPLES,
  TOOL_DESC, LABELS, LLM_BASELINE, CASES_KEY, REPEAT,
  INSTANCE_TYPE, INSTANCE_VCPUS, INSTANCE_HOURLY_USD
"""
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import boto3
from botocore.config import Config

BUCKET = os.environ.get("BUCKET", "")
PROJECT = os.environ.get("PROJECT", "")
TASK = os.environ.get("TASK", "generation").strip().lower()
INFERENCE = os.environ.get("INFERENCE_ENDPOINT", "").rstrip("/")
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL", "")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "100") or "100")
TOOL_DESC = os.environ.get("TOOL_DESC", "")
LABELS = [l.strip() for l in os.environ.get("LABELS", "").split(",") if l.strip()]
LLM_BASELINE = os.environ.get("LLM_BASELINE", "").lower() in ("1", "true", "yes")
CASES_KEY = os.environ.get("CASES_KEY", "")
REPEAT = max(1, int(os.environ.get("REPEAT", "2") or "2"))
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "")
INSTANCE_VCPUS = int(os.environ.get("INSTANCE_VCPUS", "0") or "0")
INSTANCE_HOURLY_USD = float(os.environ.get("INSTANCE_HOURLY_USD", "0") or "0")
LATENCY_SAMPLES = 30
# Bedrock calls from the report are rate limited so the report can never be
# the thing that trips an account limit.
BEDROCK_MIN_INTERVAL_S = 1.0

s3 = boto3.client("s3")
_bedrock = None
_last_bedrock_call = 0.0


def p(msg):
    print(msg, flush=True)


# ── S3 and HTTP helpers ─────────────────────────────────────────────────────────

def load_json(key, default=None):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode())
    except Exception as e:  # noqa: BLE001
        p(f"  (no {key}: {e})")
        return default


def load_jsonl(key):
    try:
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()
    except Exception as e:  # noqa: BLE001
        p(f"  (no {key}: {e})")
        return []
    out = []
    for line in body.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def http_json(path, body=None, timeout=300):
    """POST (or GET when body is None) JSON to the inference endpoint.
    Returns (json, elapsed_ms)."""
    url = f"{INFERENCE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())
    return payload, (time.perf_counter() - t0) * 1000


def bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", config=Config(read_timeout=120, retries={"max_attempts": 6, "mode": "adaptive"}))
    return _bedrock


def bedrock_text(prompt, max_tokens=64):
    """One rate-limited Bedrock call; returns the text of the reply."""
    global _last_bedrock_call
    wait = BEDROCK_MIN_INTERVAL_S - (time.time() - _last_bedrock_call)
    if wait > 0:
        time.sleep(wait)
    _last_bedrock_call = time.time()
    resp = bedrock().converse(
        modelId=BEDROCK_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0})
    return resp["output"]["message"]["content"][0]["text"].strip()


def pct(values, q):
    if not values:
        return None
    vs = sorted(values)
    idx = min(len(vs) - 1, max(0, round(q * (len(vs) - 1))))
    return round(vs[idx], 1)


def latency_summary(ms_values):
    if not ms_values:
        return {}
    return {"n": len(ms_values), "p50_ms": pct(ms_values, 0.5), "p95_ms": pct(ms_values, 0.95),
            "max_ms": round(max(ms_values), 1)}


def serving_block():
    """The node the inference pod landed on. Hourly rate is on-demand list
    price for one node, looked up by the CLI; it is capacity you pay for
    whether or not a request arrives, so it is never divided into a per-request
    figure here."""
    return {"instance_type": INSTANCE_TYPE or None, "vcpus": INSTANCE_VCPUS or None,
            "hourly_usd": round(INSTANCE_HOURLY_USD, 4) if INSTANCE_HOURLY_USD else None,
            "monthly_usd": round(INSTANCE_HOURLY_USD * 730, 2) if INSTANCE_HOURLY_USD else None}


# ── Encoder-family tasks ────────────────────────────────────────────────────────

def probe_encoder_latency(task, eval_rows):
    """Time LATENCY_SAMPLES requests against the served endpoint, using real
    held-out inputs, one at a time (the shape of a routing call)."""
    inputs = []
    for r in eval_rows:
        text = r.get("input") or r.get("query") or ""
        if text:
            inputs.append(text)
        if len(inputs) >= LATENCY_SAMPLES:
            break
    if not inputs:
        return {}
    # One warm-up request so the first measurement is not the model's first.
    try:
        _encoder_call(task, inputs[0])
    except Exception as e:  # noqa: BLE001
        p(f"  warm-up request failed: {e}")
        return {"error": str(e)}
    times = []
    for text in inputs:
        try:
            _, ms = _encoder_call(task, text)
            times.append(ms)
        except Exception as e:  # noqa: BLE001
            p(f"  request failed: {e}")
    return latency_summary(times)


def _encoder_call(task, text):
    if task == "classification":
        return http_json("/v1/chat/completions", {"model": "model", "max_tokens": 16,
                                                  "messages": [{"role": "user", "content": text}]}, timeout=30)
    if task == "embedding":
        return http_json("/embed", {"inputs": text[:8000]}, timeout=30)
    if task == "scoring":
        return http_json("/score", {"input": text}, timeout=30)
    if task == "extraction":
        return http_json("/extract", {"input": text}, timeout=30)
    raise ValueError(task)


def llm_zero_shot_baseline(eval_rows, labels):
    """The control: the frontier model, prompted with the label set, on the
    same held-out inputs the head was scored on. Exact match on the label,
    the same way the head is scored. One call per sample, rate limited."""
    rows = [r for r in eval_rows if r.get("input") and r.get("output")][:MAX_SAMPLES]
    if not rows or not labels or not BEDROCK_MODEL:
        return None
    label_list = ", ".join(labels)
    correct_flags, origins, preds = [], [], []
    for r in rows:
        prompt = (f"Classify the message into exactly one of these categories: {label_list}.\n"
                  f"Task description: {TOOL_DESC[:600]}\n\n"
                  f"MESSAGE:\n{r['input'][:3000]}\n\n"
                  "Reply with the category name only.")
        expected = r["output"].split("|")[0].strip()
        try:
            text = bedrock_text(prompt, max_tokens=16)
        except Exception as e:  # noqa: BLE001
            p(f"  baseline call failed: {e}")
            continue
        pred = next((l for l in labels if l.lower() in text.lower()), text.strip().split()[0] if text.strip() else "")
        correct_flags.append(int(pred == expected))
        origins.append(r.get("origin", "synthetic"))
        preds.append({"input": r["input"][:200], "expected": expected, "predicted": pred,
                      "correct": pred == expected, "origin": origins[-1]})
    if not correct_flags:
        return None
    by_origin = {}
    for o, c in zip(origins, correct_flags):
        g = by_origin.setdefault(o, {"n": 0, "correct": 0})
        g["n"] += 1
        g["correct"] += c
    for g in by_origin.values():
        g["accuracy"] = round(g["correct"] / g["n"], 4)
    return {"model": BEDROCK_MODEL, "n": len(correct_flags),
            "accuracy": round(sum(correct_flags) / len(correct_flags), 4),
            "by_origin": by_origin, "predictions": preds}


def encoder_report(task):
    metrics = load_json(f"models/{PROJECT}/metrics.json", default={})
    predictions = load_jsonl(f"models/{PROJECT}/eval_predictions.jsonl")
    eval_rows = load_jsonl(f"{PROJECT}/processed/eval.jsonl")
    p(f"Metrics: {'loaded' if metrics else 'missing'}; predictions: {len(predictions)}; eval rows: {len(eval_rows)}")

    p(f"Measuring endpoint latency ({LATENCY_SAMPLES} requests)...")
    latency = probe_encoder_latency(task, eval_rows) if INFERENCE else {}
    if latency.get("p50_ms") is not None:
        p(f"  p50 {latency['p50_ms']} ms, p95 {latency['p95_ms']} ms")

    llm = None
    if task == "classification" and LLM_BASELINE:
        p(f"Running the zero-shot frontier-model baseline ({min(len(eval_rows), MAX_SAMPLES)} calls, 1 per second)...")
        llm = llm_zero_shot_baseline(eval_rows, LABELS or metrics.get("classes", []))
        if llm:
            p(f"  frontier model: {llm['accuracy'] * 100:.1f}% on {llm['n']} held-out samples")

    report = {
        "project": PROJECT, "task": task, "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics, "latency": latency, "llm_baseline": llm, "serving": serving_block(),
        "predictions": predictions[:500],
    }
    items = guidance(task, metrics, llm)
    # Specific findings first; the last item is always the generic order of
    # investigation. The terminal summary prints only the findings.
    report["findings"], report["guidance"] = items[:-1], items
    return report


def guidance(task, metrics, llm=None):
    """The order to look for the cause when a number is low. Data first."""
    items = []
    if task == "classification":
        by = metrics.get("by_origin") or {}
        real, syn = by.get("real"), by.get("synthetic")
        if real and syn and real.get("accuracy") is not None and syn.get("accuracy") is not None:
            gap = syn["accuracy"] - real["accuracy"]
            if gap > 0.05:
                items.append(f"Accuracy on human-labeled held-out data is {gap * 100:.0f} points below the synthetic set. "
                             "The generated data has drifted from what users write: revise the domain description in expert.yaml "
                             "and add real seed examples before touching the model.")
        base = (metrics.get("baseline") or {}).get("accuracy")
        acc = metrics.get("accuracy")
        if base is not None and acc is not None and acc - base < 0.15:
            items.append("The head is within 15 points of the majority-class baseline. Check that the labels are separable "
                         "from the text (read the confusions) and that each class has enough examples.")
        for b in metrics.get("calibration") or []:
            if b["n"] >= 5 and b["avg_confidence"] - b["accuracy"] > 0.15:
                items.append(f"Predictions with probability {b['low']:.1f} to {b['high']:.1f} are right {b['accuracy'] * 100:.0f}% of the time "
                             f"but claim {b['avg_confidence'] * 100:.0f}%. The head is over-confident there; those are the inputs to hand-check.")
                break
        if llm and acc is not None and llm["accuracy"] - acc > 0.05:
            items.append("The frontier model scores higher on the same held-out set. The gap is what more or better training data would buy; "
                         "the frontier model's misses tell you which classes are ambiguous by definition.")
    if task == "embedding":
        t, b = metrics.get("tuned") or {}, metrics.get("baseline") or {}
        if t.get("recall@2") is not None and b.get("recall@2") is not None and t["recall@2"] - b["recall@2"] < 0.02:
            items.append("Fine-tuning did not move recall@2. The stock encoder is as good on this corpus; serve it stock, or check that "
                         "the generated (question, chunk) pairs read like real questions.")
        by = t.get("by_origin") or {}
        if by.get("real") and by.get("synthetic") and by["synthetic"]["recall@2"] - by["real"]["recall@2"] > 0.1:
            items.append("Recall on human-written questions is well below recall on generated ones. The generated questions are easier "
                         "than real ones; add real seeds to data.evaluation.labeled and revise the domain description.")
    items.append("Order of investigation when a number is low: the data (coverage and labels), then the held-out set itself, "
                 "then the prompt or head settings, and only then a different model.")
    return items


# ── Generation ──────────────────────────────────────────────────────────────────

# Published per-socket memory bandwidth and vCPUs per socket, used to estimate
# the share a node of a given size gets. These are estimates for the ceiling
# arithmetic, not measurements; the report labels them as such.
BANDWIDTH_TABLE = {
    # family prefix: (GB/s per socket, vCPUs per socket, note)
    "7g": (307, 64, "Graviton3: 8 channels DDR5-4800"),
    "8g": (537, 96, "Graviton4: 12 channels DDR5-5600"),
    "9g": (800, 192, "Graviton5: DDR5-8800; AWS quotes more than 800 GB/s aggregate"),
    "7a": (460, 96, "AMD EPYC Genoa: 12 channels DDR5-4800"),
    "7i": (307, 96, "Intel Sapphire Rapids: 8 channels DDR5-4800"),
    "8i": (410, 96, "Intel Emerald Rapids / Granite Rapids class: 8 channels DDR5-6400"),
}


def bandwidth_estimate(instance_type, vcpus):
    """(GB/s available to this node, note) or (None, reason)."""
    m = re.match(r"^[a-z]+(\d)([a-z]*)\.", instance_type or "")
    if not m:
        return None, "instance type unknown"
    gen, suffix = m.group(1), m.group(2)
    key = None
    if "g" in suffix:
        key = f"{gen}g"
    elif "a" in suffix:
        key = f"{gen}a"
    elif "i" in suffix or suffix == "":
        key = f"{gen}i"
    row = BANDWIDTH_TABLE.get(key)
    if not row:
        return None, f"no published figure in the table for {instance_type}"
    per_socket, socket_vcpus, note = row
    if not vcpus:
        return per_socket, f"{note}; full-socket figure, node size unknown"
    share = min(1.0, vcpus / socket_vcpus)
    return round(per_socket * share, 1), f"{note}; {vcpus} of {socket_vcpus} vCPUs, so about {share * 100:.0f}% of a socket"


def _chat(prompt, max_tokens=160, timeout=900):
    body = {"model": "model", "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    return http_json("/v1/chat/completions", body, timeout=timeout)


def _timings(resp):
    t = resp.get("timings") or {}
    return {"prompt_tokens": t.get("prompt_n"), "prompt_ms": round(t.get("prompt_ms", 0), 1),
            "prompt_tok_s": round(t.get("prompt_per_second", 0), 1),
            "decode_tokens": t.get("predicted_n"), "decode_ms": round(t.get("predicted_ms", 0), 1),
            "decode_tok_s": round(t.get("predicted_per_second", 0), 1)}


def serving_profile():
    """Probe llama.cpp: model facts, then a cold-prefix and a warm-prefix
    completion so the two time-to-first-token figures are both measured."""
    profile = {}
    try:
        models, _ = http_json("/v1/models", timeout=30)
        meta = (models.get("data") or [{}])[0].get("meta") or {}
        profile["model_size_bytes"] = meta.get("size")
        profile["n_params_total"] = meta.get("n_params")
        profile["n_ctx_train"] = meta.get("n_ctx_train")
    except Exception as e:  # noqa: BLE001
        p(f"  /v1/models failed: {e}")
    try:
        props, _ = http_json("/props", timeout=30)
        profile["model_path"] = (props.get("model_path") or "").split("/")[-1]
        profile["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        profile["slots"] = props.get("total_slots")
    except Exception as e:  # noqa: BLE001
        p(f"  /props failed: {e}")

    # A prompt the server has not seen (a fresh prefix), long enough to make
    # prefill visible, then the same prompt again so the prefix cache serves it.
    # The nonce at the front keeps the prefix cold even when the report re-runs
    # against a server that has already answered this prompt.
    nonce = datetime.now(timezone.utc).strftime("run %Y%m%d%H%M%S%f")
    filler = " ".join(f"Reference item {i}: the field spec.limits.cpu caps the total CPU a NodePool may provision; "
                      f"a value of 0 means no node can be launched for it." for i in range(24))
    prompt = (f"[{nonce}] Using only this reference:\n{filler}\n\nQuestion: what happens to pods that target a NodePool "
              "whose spec.limits.cpu is 0, and what is the fix? Answer in three sentences.")
    for label in ("cold", "warm"):
        try:
            resp, ms = _chat(prompt)
            t = _timings(resp)
            t["total_ms"] = round(ms, 1)
            t["ttft_ms"] = t["prompt_ms"]
            profile[label] = t
            p(f"  {label}: prompt {t['prompt_tokens']} tok in {t['prompt_ms']} ms ({t['prompt_tok_s']} tok/s), "
              f"decode {t['decode_tokens']} tok at {t['decode_tok_s']} tok/s")
        except Exception as e:  # noqa: BLE001
            p(f"  {label} completion failed: {e}")
            profile[label] = {"error": str(e)}

    gbps, note = bandwidth_estimate(INSTANCE_TYPE, INSTANCE_VCPUS)
    size = profile.get("model_size_bytes")
    ceiling = None
    if gbps and size:
        ceiling = round(gbps * 1e9 / size, 1)
    profile["ceiling"] = {
        "bandwidth_gb_s": gbps, "bandwidth_note": note,
        "bytes_per_token_assumed": size,
        "tokens_per_second_ceiling": ceiling,
        "assumption": ("bytes per token = the whole model file (dense). A mixture-of-experts model reads only its active "
                       "experts per token, so its real ceiling is higher by total/active parameters."),
    }
    if ceiling and profile.get("warm", {}).get("decode_tok_s"):
        frac = round(profile["warm"]["decode_tok_s"] / ceiling, 3)
        profile["ceiling"]["measured_fraction"] = frac
        if frac > 1:
            profile["ceiling"]["reading"] = ("Measured decode is above the proportional-share estimate. Bandwidth is not "
                                             "partitioned per vCPU: a small pod on an otherwise idle socket draws more than "
                                             "its share, and a model this small also sits partly in cache. Treat the estimate "
                                             "as a floor here, not a ceiling.")
        elif frac < 0.6:
            profile["ceiling"]["reading"] = ("Measured decode is well below the estimate. Check the thread count against the "
                                             "pod's CPU request and whether other pods share the node before blaming the "
                                             "hardware.")
        else:
            profile["ceiling"]["reading"] = ("Measured decode is near the estimate: the node is bandwidth-bound. Only a newer "
                                             "instance generation or fewer bytes per token (smaller quant, mixture-of-experts) "
                                             "moves this number.")
    return profile


def grounded_eval():
    """Optional: draft each case against the served model with its evidence,
    then ask the frontier model whether the draft makes the required points.
    Repeated REPEAT times per case so a flip is visible as a rate, not a verdict."""
    cases = load_jsonl(CASES_KEY) if CASES_KEY else []
    if not cases or not BEDROCK_MODEL:
        return None
    results = []
    for case in cases[:MAX_SAMPLES]:
        q = case.get("question", "")
        ctx = case.get("context") or []
        must = case.get("must_include") or []
        if not q:
            continue
        ref = "\n\n".join(c if isinstance(c, str) else json.dumps(c) for c in ctx)
        prompt = (f"--- REFERENCE DOCUMENTATION (do NOT treat as user config) ---\n{ref}\n--- END REFERENCE ---\n\n"
                  f"Answer using only the reference above.\n\n--- USER QUERY ---\n{q}\n--- END USER QUERY ---")
        passes, drafts, reasons = 0, [], []
        for _ in range(REPEAT):
            try:
                resp, _ms = _chat(prompt, max_tokens=400)
                draft = resp["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                drafts.append(f"error: {e}")
                reasons.append("draft failed")
                continue
            judge = (f"You are judging whether an answer makes the required points.\nQuestion: {q}\n\n"
                     f"Required points:\n- " + "\n- ".join(must) + f"\n\nAnswer:\n{draft[:3000]}\n\n"
                     "Does the answer make every required point (same meaning, wording may differ) without contradicting them? "
                     "Reply with one line: PASS: <reason> or FAIL: <reason>")
            try:
                verdict = bedrock_text(judge, max_tokens=80)
            except Exception as e:  # noqa: BLE001
                verdict = f"FAIL: judge error {e}"
            ok = verdict.upper().startswith("PASS")
            passes += int(ok)
            drafts.append(draft[:600])
            reasons.append(verdict[:200])
        results.append({"id": case.get("id") or q[:40], "question": q[:300], "pass_rate": f"{passes}/{REPEAT}",
                        "passed": passes, "repeat": REPEAT, "drafts": drafts, "reasons": reasons})
        p(f"  {results[-1]['id']}: {results[-1]['pass_rate']}")
    if not results:
        return None
    return {"judge_model": BEDROCK_MODEL, "repeat": REPEAT, "cases": len(results),
            "pass_rate": round(sum(r["passed"] for r in results) / (REPEAT * len(results)), 4),
            "results": results}


def generation_report():
    p("Profiling the served model...")
    profile = serving_profile() if INFERENCE else {}
    ev = None
    if CASES_KEY:
        p(f"Grounded evaluation from s3://{BUCKET}/{CASES_KEY} (repeat={REPEAT})...")
        ev = grounded_eval()
    findings = []
    cold = profile.get("cold") or {}
    if cold.get("ttft_ms") and cold["ttft_ms"] > 2000:
        findings.append(f"Cold time to first token is {cold['ttft_ms'] / 1000:.1f} s for a {cold.get('prompt_tokens', '?')}-token "
                        "prompt. That is prompt processing: keep retrieved context tight, warm common prefixes at start, and "
                        "stream the answer. If it still misses the budget, that is the case for an accelerator.")
    if ev and ev["pass_rate"] < 1:
        failed = [r["id"] for r in ev["results"] if r["passed"] < r["repeat"]]
        findings.append(f"{len(failed)} of {ev['cases']} grounded cases did not pass every run ({', '.join(failed[:5])}). "
                        "Check the evidence handed to the model before the model: most misses are the right chunk not "
                        "being in the context.")
    reading = (profile.get("ceiling") or {}).get("reading")
    if reading and "well below" in reading:
        findings.append(reading)
    generic = ["Decode speed is bounded by memory bandwidth divided by bytes per token; a newer instance generation raises "
               "the ceiling, more vCPUs on the same generation do not raise it proportionally."]
    return {"project": PROJECT, "task": "generation", "generated_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile, "grounded_eval": ev, "serving": serving_block(),
            "findings": findings, "guidance": findings + generic}


# ── HTML ────────────────────────────────────────────────────────────────────────

CSS = """
:root{--bg:#fff;--fg:#0f172a;--muted:#64748b;--card:#f1f5f9;--line:#e2e8f0;--sky:#0ea5e9;--amber:#f59e0b;--violet:#7c3aed;--green:#059669;--red:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#0f172a;--fg:#f1f5f9;--muted:#94a3b8;--card:#1e293b;--line:#334155}}
body{margin:0;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--fg);line-height:1.45}
main{max-width:1100px;margin:0 auto;padding:32px 24px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px;padding-top:12px;border-top:1px solid var(--line)}h3{font-size:14px;margin:18px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--muted);font-size:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.card{background:var(--card);border-radius:10px;padding:14px 16px}.card .v{font-size:26px;font-weight:700;font-family:ui-monospace,Menlo,monospace}.card .l{font-size:12px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600;font-size:12px}
code{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.ok{color:var(--green);font-weight:600}.bad{color:var(--red);font-weight:600}.warn{color:var(--amber)}
.note{background:var(--card);border-left:4px solid var(--sky);padding:10px 14px;border-radius:6px;font-size:13px;margin:12px 0}
ol.guide li{margin:6px 0}
details summary{cursor:pointer;color:var(--muted);font-size:13px}
"""


def esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fmt(v, kind="num"):
    if v is None:
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "ms":
        return f"{v:,.0f} ms"
    if kind == "usd":
        return f"${v:,.2f}"
    if isinstance(v, float):
        return f"{v:,.3f}"
    return f"{v:,}" if isinstance(v, int) else esc(v)


def card(value, label):
    return f'<div class="card"><div class="v">{value}</div><div class="l">{esc(label)}</div></div>'


def table(headers, rows):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def render_serving(s):
    rows = []
    if s.get("instance_type"):
        rows.append(["Instance type", esc(s["instance_type"])])
    if s.get("vcpus"):
        rows.append(["vCPUs", fmt(s["vcpus"])])
    if s.get("hourly_usd"):
        rows.append(["On-demand rate, one node", f"{fmt(s['hourly_usd'], 'usd')}/hour, about {fmt(s['monthly_usd'], 'usd')}/month"])
    if not rows:
        return "<p class='sub'>Node details were not available to the report Job.</p>"
    return table(["", ""], rows) + ("<p class='note'>The node is fixed capacity: it costs the same per hour whether one request arrives or a thousand. "
                                   "Dividing it into a per-request figure needs your request rate, which the report does not know. "
                                   "Replicas add throughput linearly and do not change the latency of one request.</p>")


def render_guidance(items):
    return "<ol class='guide'>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ol>"


def render_encoder(rep):
    m, task = rep["metrics"], rep["task"]
    lat, llm = rep.get("latency") or {}, rep.get("llm_baseline")
    parts = []
    # Summary cards
    cards = []
    if task == "classification":
        cards.append(card(fmt(m.get("accuracy"), "pct"), f"exact-match accuracy, {m.get('total', 0)} held-out"))
        base = m.get("baseline") or {}
        if base.get("accuracy") is not None:
            cards.append(card(fmt(base["accuracy"], "pct"), f"majority-class baseline (always '{base.get('label')}')"))
        if llm:
            cards.append(card(fmt(llm["accuracy"], "pct"), f"frontier model zero-shot, {llm['n']} samples"))
    elif task == "embedding":
        t, b = m.get("tuned") or {}, m.get("baseline") or {}
        cards.append(card(fmt(t.get("recall@2"), "pct"), "recall@2, tuned encoder"))
        cards.append(card(fmt(b.get("recall@2"), "pct"), "recall@2, stock encoder"))
        cards.append(card(fmt(t.get("mrr")), "MRR, tuned"))
    elif task == "scoring":
        cards.append(card(fmt(m.get("mae")), "MAE"))
        cards.append(card(fmt(m.get("baseline_mae")), "MAE, predict-the-mean baseline"))
        cards.append(card(fmt(m.get("r2")), "R squared"))
    elif task == "extraction":
        cards.append(card(fmt(m.get("f1"), "pct"), "F1"))
        cards.append(card(fmt(m.get("baseline_f1"), "pct"), "F1, gazetteer baseline"))
    if lat.get("p50_ms") is not None:
        cards.append(card(fmt(lat["p50_ms"], "ms"), f"endpoint latency p50 ({lat['n']} requests)"))
    parts.append('<div class="cards">' + "".join(cards) + "</div>")

    # Quality
    parts.append("<h2>Quality</h2>")
    if task == "classification":
        by = m.get("by_origin") or {}
        if by:
            parts.append("<h3>Held-out set by origin</h3>")
            rows = [[esc(o), fmt(g["n"]), fmt(g["correct"]), fmt(g["accuracy"], "pct")] for o, g in sorted(by.items())]
            if llm and llm.get("by_origin"):
                for r, (o, g) in zip(rows, sorted(by.items())):
                    lg = llm["by_origin"].get(o)
                    r.append(fmt(lg["accuracy"], "pct") if lg else "n/a")
                parts.append(table(["origin", "n", "correct", "head accuracy", "frontier model"], rows))
            else:
                parts.append(table(["origin", "n", "correct", "accuracy"], rows))
            if "real" not in by:
                parts.append("<p class='note'>All held-out records are synthetic. Add human-labeled examples under "
                             "<code>data.evaluation.labeled</code> to see how the head does on what users actually write.</p>")
        pc = m.get("per_class") or {}
        if pc:
            parts.append("<h3>Per class</h3>")
            rows = [[esc(c), fmt(v["precision"]), fmt(v["recall"]), fmt(v["f1"])]
                    for c, v in sorted(pc.items(), key=lambda kv: kv[1]["f1"])]
            parts.append(table(["class", "precision", "recall", "F1"], rows))
        conf = m.get("confusions") or []
        if conf:
            parts.append("<h3>Most frequent confusions</h3>")
            parts.append(table(["expected", "predicted", "count"],
                               [[esc(c["expected"]), esc(c["predicted"]), fmt(c["count"])] for c in conf]))
        cal = m.get("calibration") or []
        if cal:
            parts.append("<h3>Reliability of the head's probability</h3>")
            rows = []
            for b in cal:
                gap = b["avg_confidence"] - b["accuracy"]
                cls = "bad" if gap > 0.15 and b["n"] >= 5 else ""
                rows.append([f"{b['low']:.1f} to {b['high']:.1f}", fmt(b["n"]), fmt(b["accuracy"], "pct"),
                             f"<span class='{cls}'>{fmt(b['avg_confidence'], 'pct')}</span>"])
            parts.append(table(["predicted probability", "n", "accuracy", "average claimed"], rows))
            parts.append("<p class='sub'>A well-calibrated head has accuracy close to the claimed probability in every row. "
                         "Rows where the claim is far above the accuracy are where to hand-check predictions.</p>")
    elif task == "embedding":
        t, b = m.get("tuned") or {}, m.get("baseline") or {}
        ks = [k for k in ("recall@1", "recall@2", "recall@5", "recall@10", "mrr") if k in t]
        parts.append("<h3>Stock versus tuned encoder</h3>")
        parts.append(table(["metric", "stock", "tuned", "change"],
                           [[k, fmt(b.get(k)), fmt(t.get(k)),
                             (f"{(t[k] - b[k]):+.3f}" if b.get(k) is not None and t.get(k) is not None else "")] for k in ks]))
        by = t.get("by_origin") or {}
        if by:
            parts.append("<h3>Tuned encoder by held-out origin</h3>")
            parts.append(table(["origin", "queries", "recall@2", "recall@5", "MRR"],
                               [[esc(o), fmt(g["eval_queries"]), fmt(g.get("recall@2"), "pct"),
                                 fmt(g.get("recall@5"), "pct"), fmt(g.get("mrr"))] for o, g in sorted(by.items())]))
        else:
            parts.append("<p class='note'>All held-out queries are synthetic. Add human-written (query, gold chunk) pairs under "
                         "<code>data.evaluation.labeled</code> to measure retrieval on real questions.</p>")
        parts.append(f"<p class='sub'>Corpus of {fmt(m.get('corpus_size'))} chunks, {fmt(m.get('eval_queries'))} held-out queries, "
                     f"{fmt(m.get('epochs'))} epochs, {fmt(m.get('train_seconds'))} s of training on CPU.</p>")
    elif task == "scoring":
        parts.append(table(["metric", "value"], [["MAE", fmt(m.get("mae"))], ["RMSE", fmt(m.get("rmse"))],
                                                 ["R squared", fmt(m.get("r2"))], ["correlation", fmt(m.get("correlation"))],
                                                 ["baseline MAE (predict the mean)", fmt(m.get("baseline_mae"))]]))
    elif task == "extraction":
        pe = m.get("per_entity") or {}
        parts.append(table(["entity", "precision", "recall", "F1"],
                           [[esc(e), fmt(v["precision"]), fmt(v["recall"]), fmt(v["f1"])] for e, v in sorted(pe.items())]))

    # Predictions
    preds = rep.get("predictions") or []
    if preds:
        parts.append("<h2>Held-out predictions</h2>")
        if task == "classification":
            preds = sorted(preds, key=lambda r: (r.get("correct", True), -r.get("probability", 0)))
            rows = [[esc(r.get("input", "")[:200]), f"<code>{esc(r.get('expected'))}</code>", f"<code>{esc(r.get('predicted'))}</code>",
                     fmt(r.get("probability"), "pct"), esc(r.get("origin", "")),
                     "<span class='ok'>yes</span>" if r.get("correct") else "<span class='bad'>no</span>"] for r in preds]
            parts.append("<details><summary>Show all rows (incorrect first)</summary>" +
                         table(["input", "expected", "predicted", "probability", "origin", "correct"], rows) + "</details>")
        elif task == "embedding":
            preds = sorted(preds, key=lambda r: (r.get("rank_tuned") or 999), reverse=True)
            rows = [[esc(r.get("query", "")[:200]), esc(r.get("origin", "")), fmt(r.get("rank_stock")) if r.get("rank_stock") else "> 10",
                     fmt(r.get("rank_tuned")) if r.get("rank_tuned") else "> 10", esc(r.get("positive", "")[:120])] for r in preds]
            parts.append("<details><summary>Show all queries (worst tuned rank first)</summary>" +
                         table(["query", "origin", "rank stock", "rank tuned", "gold chunk"], rows) + "</details>")

    # Latency and serving
    parts.append("<h2>Latency</h2>")
    if lat.get("p50_ms") is not None:
        parts.append(table(["", "value"], [["requests", fmt(lat["n"])], ["p50", fmt(lat["p50_ms"], "ms")],
                                           ["p95", fmt(lat["p95_ms"], "ms")], ["max", fmt(lat["max_ms"], "ms")]]))
        parts.append("<p class='sub'>Measured one request at a time against the served endpoint from inside the cluster, "
                     "on real held-out inputs, after one warm-up request.</p>")
    else:
        parts.append(f"<p class='sub'>Not measured: {esc(lat.get('error', 'endpoint unavailable'))}</p>")
    parts.append("<h2>Serving</h2>")
    parts.append(render_serving(rep.get("serving") or {}))
    parts.append("<h2>When a number is low</h2>")
    parts.append(render_guidance(rep.get("guidance") or []))
    return "".join(parts)


def render_generation(rep):
    pr = rep.get("profile") or {}
    cold, warm, ceil = pr.get("cold") or {}, pr.get("warm") or {}, pr.get("ceiling") or {}
    parts = []
    cards = []
    if warm.get("decode_tok_s"):
        cards.append(card(f"{warm['decode_tok_s']:.1f}", "decode tokens per second, warm"))
    if cold.get("ttft_ms") is not None:
        cards.append(card(fmt(cold["ttft_ms"], "ms"), f"time to first token, cold prefix ({cold.get('prompt_tokens')} prompt tokens)"))
    if warm.get("ttft_ms") is not None:
        cards.append(card(fmt(warm["ttft_ms"], "ms"), "time to first token, warm prefix"))
    if ceil.get("tokens_per_second_ceiling"):
        cards.append(card(f"{ceil['tokens_per_second_ceiling']:.0f}", "tokens per second ceiling (estimate)"))
    parts.append('<div class="cards">' + "".join(cards) + "</div>")

    parts.append("<h2>Model</h2>")
    rows = [["file", f"<code>{esc(pr.get('model_path') or 'n/a')}</code>"],
            ["size on disk", f"{pr['model_size_bytes'] / 1e9:.2f} GB" if pr.get("model_size_bytes") else "n/a"],
            ["parameters (total)", f"{pr['n_params_total'] / 1e9:.1f} B" if pr.get("n_params_total") else "n/a"],
            ["context (served / trained)", f"{fmt(pr.get('n_ctx'))} / {fmt(pr.get('n_ctx_train'))}"],
            ["slots", fmt(pr.get("slots"))]]
    parts.append(table(["", ""], rows))

    parts.append("<h2>Prefill and decode</h2>")
    rows = []
    for label, t in (("cold prefix", cold), ("warm prefix", warm)):
        if t.get("error"):
            rows.append([label, f"<span class='bad'>{esc(t['error'])}</span>", "", "", "", ""])
        else:
            rows.append([label, fmt(t.get("prompt_tokens")), fmt(t.get("prompt_ms"), "ms"), f"{t.get('prompt_tok_s', 0):.0f}",
                         fmt(t.get("decode_tokens")), f"{t.get('decode_tok_s', 0):.1f}"])
    parts.append(table(["request", "prompt tokens", "prompt time", "prompt tok/s", "decode tokens", "decode tok/s"], rows))
    parts.append("<p class='sub'>Prefill (the prompt) is compute-bound and parallel; decode (the answer) reads the active weights "
                 "once per token and is bound by memory bandwidth. The second request repeats the first so the prompt-prefix "
                 "cache serves it: that difference is what warming a prompt at startup buys.</p>")

    parts.append("<h2>Bandwidth ceiling</h2>")
    rows = [["node", esc(rep.get("serving", {}).get("instance_type") or "unknown")],
            ["bandwidth available (estimate)", f"{ceil.get('bandwidth_gb_s') or 'n/a'} GB/s"],
            ["basis", esc(ceil.get("bandwidth_note") or "")],
            ["bytes read per token (assumed)", f"{ceil['bytes_per_token_assumed'] / 1e9:.2f} GB" if ceil.get("bytes_per_token_assumed") else "n/a"],
            ["ceiling", f"{ceil['tokens_per_second_ceiling']:.0f} tokens/s" if ceil.get("tokens_per_second_ceiling") else "n/a"],
            ["measured warm decode as a fraction of the ceiling", fmt(ceil.get("measured_fraction"))]]
    parts.append(table(["", ""], rows))
    if ceil.get("reading"):
        parts.append(f"<p>{esc(ceil['reading'])}</p>")
    parts.append(f"<p class='note'>{esc(ceil.get('assumption', ''))} The bandwidth figure is a published per-socket number scaled by "
                 "this node's share of the socket; it is an estimate for the arithmetic, not a measurement.</p>")

    ev = rep.get("grounded_eval")
    parts.append("<h2>Grounded evaluation</h2>")
    if ev:
        parts.append(f"<p>{fmt(ev['pass_rate'], 'pct')} of drafts made every required point, {ev['cases']} cases, "
                     f"each run {ev['repeat']} times, judged by <code>{esc(ev['judge_model'])}</code>.</p>")
        rows = [[esc(r["id"]), esc(r["question"][:160]), r["pass_rate"], esc((r["reasons"] or [""])[-1][:160])] for r in ev["results"]]
        parts.append(table(["case", "question", "passes", "last verdict"], rows))
        parts.append("<p class='sub'>A case that passes some repeats and fails others is judge noise, not a verdict; "
                     "re-run it before treating it as a regression.</p>")
    else:
        parts.append("<p class='sub'>Not run. Point <code>report.cases</code> in the expert config at a JSONL file of "
                     "{question, context, must_include} cases to score grounded answers with a frontier-model judge.</p>")

    parts.append("<h2>Serving</h2>")
    parts.append(render_serving(rep.get("serving") or {}))
    parts.append("<h2>When a number is low</h2>")
    parts.append(render_guidance(rep.get("guidance") or []))
    return "".join(parts)


def render_html(rep):
    body = render_generation(rep) if rep["task"] == "generation" else render_encoder(rep)
    when = rep.get("generated_at", "")[:19].replace("T", " ")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Slemify report: {esc(PROJECT)}</title>"
            f"<style>{CSS}</style></head><body><main>"
            f"<h1>{esc(PROJECT)}</h1><p class='sub'>task <code>{esc(rep['task'])}</code>, generated {esc(when)} UTC. "
            "Numbers are measured; nothing here is a verdict.</p>"
            f"{body}</main></body></html>")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    p(f"=== Slemify report: {PROJECT} ({TASK}) ===")
    if not BUCKET or not PROJECT:
        raise SystemExit("BUCKET and PROJECT are required")
    if TASK == "generation":
        rep = generation_report()
    elif TASK in ("classification", "embedding", "scoring", "extraction"):
        rep = encoder_report(TASK)
    else:
        raise SystemExit(f"unknown task {TASK}")
    html = render_html(rep)
    s3.put_object(Bucket=BUCKET, Key=f"{PROJECT}/report/report.json",
                  Body=json.dumps(rep, indent=2, default=str).encode(), ContentType="application/json")
    s3.put_object(Bucket=BUCKET, Key=f"{PROJECT}/report/report.html",
                  Body=html.encode(), ContentType="text/html")
    p(f"Report: s3://{BUCKET}/{PROJECT}/report/report.html")
    p("=== Done ===")


if __name__ == "__main__":
    main()
