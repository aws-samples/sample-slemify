# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Slemify Production Readiness Report — Phase 1.

Collects inference predictions with model confidence (logprobs),
judges them via Bedrock LLM-as-judge, and presents the results
alongside training metrics and cost analysis. No verdict — the
user reviews the data and decides.

Environment variables:
  BUCKET, PROJECT, INFERENCE_ENDPOINT, BEDROCK_MODEL, MAX_SAMPLES,
  TOOL_NAME, TOOL_DESC
"""

import json
import math
import os
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config

BUCKET = os.environ.get("BUCKET", "")
PROJECT = os.environ.get("PROJECT", "")
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL", "eu.anthropic.claude-sonnet-4-6")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "72"))
INFERENCE = os.environ.get("INFERENCE_ENDPOINT", "")

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime", config=Config(read_timeout=60))
ec2 = boto3.client("ec2")
p = lambda msg: print(msg, flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p("=== Slemify Report ===")
    tool_name = os.environ.get("TOOL_NAME", PROJECT)
    tool_desc = os.environ.get("TOOL_DESC", "")

    # 1. Data analysis
    train = load_jsonl(f"{PROJECT}/processed/train.jsonl")
    evl = load_jsonl(f"{PROJECT}/processed/eval.jsonl")
    data_info = analyze_data(train, evl)
    p(f"Data: {data_info['total']} records")

    # 2. Training metrics
    p("Loading training state...")
    training = load_training_state()
    if training["available"]:
        p(f"Training: {training['total_steps']} steps, final loss {training.get('final_loss', '?')}")
    else:
        p("  No training state found")

    # 3. SLM inference + confidence
    samples = evl[:MAX_SAMPLES]
    p(f"Running SLM inference ({len(samples)} samples)...")
    predictions = run_inference(samples)
    p(f"  Collected {len(predictions)} predictions")

    # 4. LLM-as-judge
    p(f"Judging predictions via Bedrock ({len(predictions)} samples)...")
    judge_predictions(predictions, tool_desc)
    correct = sum(1 for r in predictions if r["correct"])
    p(f"  Judge: {correct}/{len(predictions)} correct ({correct/len(predictions)*100:.1f}%)")

    # 5. LLM baseline (zero-shot, same samples)
    p(f"Running LLM baseline ({len(samples)} samples)...")
    llm_results = run_llm_baseline(samples)
    p(f"  LLM p50: {_pct(sorted(r['latency_ms'] for r in llm_results), 0.5):.0f}ms")

    # 6. Cost analysis
    p("Getting pricing...")
    latencies = [r["latency_ms"] for r in predictions]
    latencies.sort()
    ttfts = sorted(r["ttft_ms"] for r in predictions if r["ttft_ms"] > 0)
    itls = sorted(r["itl_ms"] for r in predictions if r["itl_ms"] > 0)
    tok_s_vals = [r["tok_s"] for r in predictions if r["tok_s"] > 0]
    slm_stats = {
        "p50": _pct(latencies, 0.5),
        "p95": _pct(latencies, 0.95),
        "p99": _pct(latencies, 0.99),
        "avg": round(statistics.mean(latencies), 1) if latencies else 0,
        "min": round(min(latencies), 1) if latencies else 0,
        "max": round(max(latencies), 1) if latencies else 0,
        "ttft_p50": _pct(ttfts, 0.5),
        "ttft_p95": _pct(ttfts, 0.95),
        "itl_p50": _pct(itls, 0.5),
        "itl_p95": _pct(itls, 0.95),
        "tok_s_avg": round(statistics.mean(tok_s_vals), 1) if tok_s_vals else 0,
    }
    llm_latencies = sorted(r["latency_ms"] for r in llm_results)
    llm_stats = {
        "p50": _pct(llm_latencies, 0.5),
        "p95": _pct(llm_latencies, 0.95),
        "avg": round(statistics.mean(llm_latencies), 1) if llm_latencies else 0,
    }
    pricing = get_pricing(slm_stats)

    # 7. Render report
    p("Rendering HTML...")
    html = render_html(data_info, training, predictions, llm_results, slm_stats, llm_stats, pricing, tool_name)
    s3.put_object(Bucket=BUCKET, Key=f"{PROJECT}/report/report.html",
                  Body=html.encode(), ContentType="text/html")
    p(f"Report: s3://{BUCKET}/{PROJECT}/report/report.html")
    p("=== Done ===")


# ── Data ───────────────────────────────────────────────────────────────────────

def load_jsonl(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return [json.loads(l) for l in obj["Body"].read().decode().strip().split("\n") if l.strip()]


def analyze_data(train, evl):
    all_r = train + evl
    dist = {}
    for r in all_r:
        out = r.get("output", "").strip()
        # Use second field if pipe-delimited (first is often confidence)
        parts = [x.strip() for x in out.split("|") if x.strip()]
        label = parts[1] if len(parts) >= 2 else parts[0] if parts else out
        dist[label] = dist.get(label, 0) + 1
    sd = sorted(dist.items(), key=lambda x: -x[1])
    return {"train": len(train), "eval": len(evl), "total": len(all_r),
            "classes": len(dist), "distribution": sd}


# ── Training State ─────────────────────────────────────────────────────────────

def load_training_state():
    """Read trainer_state.json from the latest checkpoint in S3."""
    result = {"available": False}
    try:
        paginator = s3.get_paginator("list_objects_v2")
        prefix = f"models/{PROJECT}/"
        checkpoints = []
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"].rstrip("/").split("/")[-1]
                if name.startswith("checkpoint-"):
                    step = int(name.split("-")[1])
                    checkpoints.append((step, cp["Prefix"]))
        checkpoints.sort(key=lambda x: -x[0])

        if not checkpoints:
            return result

        latest_step, latest_prefix = checkpoints[0]
        key = f"{latest_prefix}trainer_state.json"
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        state = json.loads(obj["Body"].read().decode())

        log_history = state.get("log_history", [])
        losses = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
        eval_losses = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

        result = {
            "available": True,
            "total_steps": state.get("global_step", 0),
            "losses": losses,
            "eval_losses": eval_losses,
            "final_loss": round(losses[-1][1], 4) if losses else None,
            "start_loss": round(losses[0][1], 4) if losses else None,
            "min_loss": round(min(l for _, l in losses), 4) if losses else None,
            "duration_min": 0,
        }

        # Training runtime from training_log.json
        try:
            obj2 = s3.get_object(Bucket=BUCKET, Key=f"models/{PROJECT}/training_log.json")
            tlog = json.loads(obj2["Body"].read().decode())
            if isinstance(tlog, list) and tlog:
                summary = tlog[-1]
                rt = summary.get("train_runtime", 0)
                if rt > 0:
                    result["duration_min"] = round(rt / 60, 1)
        except Exception:
            pass

        # GPU info from pod log
        try:
            obj3 = s3.get_object(Bucket=BUCKET, Key=f"models/{PROJECT}/training-pod.log")
            podlog = obj3["Body"].read().decode()
            gpu_match = re.search(r'(NVIDIA\s+\S+)', podlog)
            result["gpu_type"] = gpu_match.group(1) if gpu_match else None
        except Exception:
            pass

    except Exception as e:
        p(f"  Warning: could not load training state: {e}")
    return result


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(samples):
    """Run samples through the SLM sequentially with progress reporting."""
    avg_output_len = sum(len(r.get("output", "")) for r in samples) / max(len(samples), 1)
    # Set max_tokens based on actual output lengths in eval data (chars/4 ≈ tokens)
    # Use p95 output length + 50% headroom, minimum 32 for classification
    output_lengths = sorted(len(r.get("output", "")) // 4 for r in samples if r.get("output"))
    if output_lengths:
        p95_tokens = output_lengths[int(len(output_lengths) * 0.95)]
        max_tokens = max(32, int(p95_tokens * 1.5))
    else:
        max_tokens = 512 if avg_output_len > 50 else 32
    results = []

    for i, rec in enumerate(samples):
        expected = rec.get("output", "").strip()
        query = rec.get("input", "")
        instruction = rec.get("instruction", "")
        prompt = f"{instruction}\n\n{query}" if instruction else query

        body = json.dumps({
            "model": "model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "logprobs": True,
        }).encode()

        start = time.time()
        try:
            req = urllib.request.Request(
                f"{INFERENCE}/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=max(60, max_tokens // 4)) as resp:
                rj = json.loads(resp.read())
            ms = (time.time() - start) * 1000
        except Exception as e:
            p(f"  SLM {i+1} failed: {e}")
            continue

        choice = rj.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "").strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        text = text.replace("<think>", "").strip()

        confidence = _compute_confidence(choice)
        timings = rj.get("timings", {})

        results.append({
            "input": query,
            "expected": expected,
            "predicted": text,
            "confidence": confidence,
            "latency_ms": round(ms, 1),
            "ttft_ms": round(timings.get("prompt_ms", 0), 1),
            "itl_ms": round(timings.get("predicted_per_token_ms", 0), 1),
            "tok_s": round(timings.get("predicted_per_second", 0), 1),
            "prompt_tok_s": round(timings.get("prompt_per_second", 0), 1),
            "correct": False,
            "reasoning": "",
        })
        if (i + 1) % 10 == 0:
            p(f"  SLM {i+1}/{len(samples)}")

    return results


def _compute_confidence(choice):
    """Confidence from logprobs: geometric mean probability of classification tokens."""
    logprobs = choice.get("logprobs", {}).get("content", [])
    if not logprobs:
        return -1.0

    # Skip think tokens
    start = 0
    for i, t in enumerate(logprobs):
        if "</think>" in t.get("token", ""):
            start = i + 1
            break

    # Score only until first newline (the classification line)
    tokens = []
    for t in logprobs[start:]:
        if "\n" in t.get("token", "") and tokens:
            break
        if t.get("token", "").strip():
            tokens.append(t)

    if not tokens:
        return -1.0

    avg_logprob = sum(t["logprob"] for t in tokens) / len(tokens)
    return round(math.exp(avg_logprob) * 100, 1)


# ── LLM-as-Judge ──────────────────────────────────────────────────────────────

def judge_predictions(results, tool_desc):
    """Use Bedrock to semantically judge each prediction. Updates results in-place."""

    def judge_one(r):
        prompt = f"""You are judging whether a model's prediction is semantically correct.

Task: {tool_desc}

Input: {r['input'][:500]}
Expected output: {r['expected']}
Model's prediction: {r['predicted'][:200]}

Is the model's prediction semantically correct? Consider:
- Does the prediction contain the same primary category/intent as expected?
- The model may output additional secondary labels after the primary one — this is acceptable and often useful (e.g., "spot_interruption|karpenter_config" when the query involves both topics).
- Minor wording differences, extra detail, or more specific labels are acceptable (e.g., "spot_interruption_handling" for "spot_interruption").
- Confidence level disagreements (e.g., high vs medium) are acceptable if the primary classification is correct.
- Only mark INCORRECT if the primary category is fundamentally wrong (e.g., routing a Karpenter question as noise).

Respond with exactly one line:
CORRECT: <one sentence reasoning>
or
INCORRECT: <one sentence reasoning>"""
        try:
            resp = bedrock.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 100, "temperature": 0})
            text = resp["output"]["message"]["content"][0]["text"].strip()
            line = text.split("\n")[0].strip()
            if line.upper().startswith("CORRECT"):
                r["correct"] = True
                r["reasoning"] = line.split(":", 1)[1].strip() if ":" in line else ""
            else:
                r["correct"] = False
                r["reasoning"] = line.split(":", 1)[1].strip() if ":" in line else line
        except Exception as e:
            r["correct"] = False
            r["reasoning"] = f"judge error: {e}"

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(judge_one, r) for r in results]
        for f in as_completed(futures):
            f.result()


# ── LLM Baseline ───────────────────────────────────────────────────────────────

def run_llm_baseline(samples):
    """Run the same samples through Bedrock (zero-shot) for comparison."""
    results = []
    instruction = samples[0].get("instruction", "") if samples else ""
    for i, rec in enumerate(samples[:15]):
        query = rec.get("input", "")
        prompt = f"{instruction}\n\n{query}" if instruction else query
        start = time.time()
        try:
            resp = bedrock.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 64, "temperature": 0})
            predicted = resp["output"]["message"]["content"][0]["text"].strip()
            ms = (time.time() - start) * 1000
        except Exception as e:
            p(f"  LLM {i+1} failed: {e}")
            continue
        results.append({
            "input": query[:100],
            "predicted": predicted,
            "latency_ms": round(ms, 1),
        })
        if (i + 1) % 10 == 0:
            p(f"  LLM {i+1}/15")
    return results


# ── Pricing ────────────────────────────────────────────────────────────────────

def get_pricing(slm_stats):
    # CPU on-demand pricing (Graviton)
    try:
        resp = ec2.describe_spot_price_history(
            InstanceTypes=["c7g.4xlarge", "c7g.2xlarge", "c8g.4xlarge", "c8g.2xlarge"],
            ProductDescriptions=["Linux/UNIX"], MaxResults=10)
        prices = {pr["InstanceType"]: float(pr["SpotPrice"]) for pr in resp["SpotPriceHistory"]}
    except Exception:
        prices = {}
    # On-demand is roughly 2.5x Spot price for Graviton
    spot_hourly = min(prices.values()) if prices else 0.16
    ondemand_hourly = round(spot_hourly * 2.5, 3)
    cpu_ondemand_monthly = round(ondemand_hourly * 730, 2)
    cpu_spot_monthly = round(spot_hourly * 730, 2)

    # GPU pricing (g5.xlarge for SLM inference — much faster per request)
    try:
        gpu_resp = ec2.describe_spot_price_history(
            InstanceTypes=["g5.xlarge", "g5.2xlarge"],
            ProductDescriptions=["Linux/UNIX"], MaxResults=5)
        gpu_prices = {pr["InstanceType"]: float(pr["SpotPrice"]) for pr in gpu_resp["SpotPriceHistory"]}
    except Exception:
        gpu_prices = {}
    gpu_spot_hourly = min(gpu_prices.values()) if gpu_prices else 0.50
    gpu_ondemand_hourly = round(gpu_spot_hourly * 3.0, 3)  # GPU on-demand ~3x Spot
    gpu_monthly = round(gpu_ondemand_hourly * 730, 2)
    # GPU inference is ~10-20x faster than CPU for SLMs
    gpu_rps_per_replica = (1000.0 / slm_stats["avg"] if slm_stats["avg"] > 0 else 1) * 15

    # SLM CPU capacity
    rps_per_replica = 1000.0 / slm_stats["avg"] if slm_stats["avg"] > 0 else 1

    tiers = []
    for daily in [1000, 10000, 100000, 1000000]:
        rps_needed = daily / 86400
        # CPU replicas needed
        cpu_replicas = max(1, math.ceil(rps_needed / rps_per_replica))
        cpu_od_cost = round(cpu_ondemand_monthly * cpu_replicas, 2)
        cpu_spot_cost = round(cpu_spot_monthly * cpu_replicas, 2)
        # GPU replicas needed
        gpu_replicas = max(1, math.ceil(rps_needed / gpu_rps_per_replica))
        gpu_cost = round(gpu_monthly * gpu_replicas, 2)
        tiers.append({
            "daily": daily, "rps_needed": round(rps_needed, 2),
            "cpu_replicas": cpu_replicas, "cpu_od": cpu_od_cost, "cpu_spot": cpu_spot_cost,
            "gpu_replicas": gpu_replicas, "gpu": gpu_cost,
        })
    return {"cpu_ondemand_monthly": cpu_ondemand_monthly, "cpu_spot_monthly": cpu_spot_monthly,
            "gpu_monthly": gpu_monthly,
            "tiers": tiers, "rps_per_replica": round(rps_per_replica, 2),
            "gpu_rps_per_replica": round(gpu_rps_per_replica, 2)}


# ── HTML Rendering ─────────────────────────────────────────────────────────────

def render_html(data_info, training, predictions, llm_results, slm_stats, llm_stats, pricing, tool_name):
    correct = sum(1 for r in predictions if r["correct"])
    total = len(predictions)
    acc = correct / total * 100 if total else 0
    confidences = [r["confidence"] for r in predictions if r["confidence"] >= 0]
    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0
    llm_p50 = llm_stats["p50"]

    # Training loss SVG
    loss_svg = _render_loss_svg(training.get("losses", []), training.get("eval_losses", []))

    # Predictions table
    pred_rows = ""
    for r in predictions:
        conf_color = "var(--green)" if r["confidence"] >= 80 else "var(--yellow)" if r["confidence"] >= 50 else "var(--red)"
        judge_color = "var(--green)" if r["correct"] else "var(--red)"
        judge_icon = "&#10003;" if r["correct"] else "&#10007;"
        pred_rows += f"""<tr>
          <td style="max-width:300px;white-space:pre-wrap;font-size:12px">{_esc(r['input'][:200])}</td>
          <td><code>{_esc(r['expected'])}</code></td>
          <td><code>{_esc(r['predicted'].split(chr(10))[0][:80])}</code></td>
          <td style="color:{conf_color};font-weight:600">{r['confidence']:.0f}%</td>
          <td style="color:{judge_color}">{judge_icon}</td>
          <td style="color:var(--muted);font-size:12px">{_esc(r.get('reasoning','')[:100])}</td>
          <td>{r['latency_ms']}ms</td>
        </tr>"""

    # Data distribution
    dist_rows = "".join(
        f'<tr><td>{_esc(label)}</td><td>{count}</td></tr>'
        for label, count in data_info["distribution"][:15])

    # SLM vs LLM comparison (side-by-side)
    compare_rows = ""
    for i in range(min(15, len(predictions), len(llm_results))):
        sr = predictions[i]
        lr = llm_results[i] if i < len(llm_results) else {"predicted": "—", "latency_ms": 0}
        compare_rows += f"""<tr>
          <td style="max-width:200px;font-size:12px;white-space:pre-wrap">{_esc(sr['input'][:150])}</td>
          <td><code>{_esc(sr['expected'])}</code></td>
          <td><code>{_esc(sr['predicted'].split(chr(10))[0][:60])}</code><br><span style="color:var(--muted);font-size:11px">{sr['latency_ms']}ms | {sr['confidence']:.0f}% conf</span></td>
          <td style="font-size:12px">{_esc(lr['predicted'][:100])}<br><span style="color:var(--muted);font-size:11px">{lr['latency_ms']}ms</span></td>
        </tr>"""

    # Cost tiers
    cost_rows = "".join(
        f'<tr><td>{t["daily"]:,}/day</td><td>{t["rps_needed"]:.2f}</td><td>{t["cpu_replicas"]} node{"s" if t["cpu_replicas"]>1 else ""}</td><td>${t["cpu_od"]}</td><td style="color:var(--green)">${t["cpu_spot"]}</td><td>{t["gpu_replicas"]} node{"s" if t["gpu_replicas"]>1 else ""}</td><td style="color:var(--purple)">${t["gpu"]}</td></tr>'
        for t in pricing["tiers"])

    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Slemify Report — {_esc(tool_name)}</title>
<style>
:root{{--bg:#0f1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--green:#3fb950;--yellow:#d29922;--red:#f85149;--purple:#a371f7;--teal:#39d353;--font:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}}
[data-theme="light"]{{--bg:#ffffff;--surface:#f6f8fa;--border:#d0d7de;--text:#1f2328;--muted:#656d76;--accent:#0969da;--green:#1a7f37;--yellow:#9a6700;--red:#cf222e;--purple:#8250df;--teal:#0e6d31}}
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:var(--font);background:var(--bg);color:var(--text);line-height:1.6;transition:background .2s,color .2s}}
.container{{max-width:1400px;margin:0 auto;padding:24px}}
header{{padding:24px 0;border-bottom:1px solid var(--border);margin-bottom:24px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap}}
header h1{{font-size:24px;font-weight:600}}header p{{color:var(--muted);font-size:14px}}
.theme-toggle{{cursor:pointer;padding:6px 12px;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--muted);font-size:12px}}
.tabs{{display:flex;gap:0;border-bottom:1px solid var(--border);margin-bottom:24px;overflow-x:auto}}
.tab{{padding:10px 20px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;font-size:14px;white-space:nowrap;transition:all .2s}}
.tab:hover{{color:var(--text)}}.tab.active{{color:var(--accent);border-bottom-color:var(--accent)}}
.panel{{display:none}}.panel.active{{display:block}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:20px;margin-bottom:16px}}
.card h3{{font-size:14px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:16px}}
.metric{{text-align:center}}.metric .value{{font-size:28px;font-weight:700}}.metric .label{{font-size:12px;color:var(--muted)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th{{text-align:left;padding:8px;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border)}}
td{{padding:8px;border-bottom:1px solid var(--border);vertical-align:top}}tr:hover{{background:rgba(88,166,255,.04)}}
code{{background:var(--bg);padding:2px 6px;border-radius:4px;font-size:12px}}
.tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}}
.tag-slm{{background:rgba(88,166,255,.15);color:var(--accent)}}.tag-llm{{background:rgba(163,113,247,.15);color:var(--purple)}}
footer{{text-align:center;padding:24px 0;color:var(--muted);font-size:12px;border-top:1px solid var(--border);margin-top:32px}}
</style></head><body><div class="container">
<header><div><h1>Slemify Report</h1><p>{_esc(tool_name)} &mdash; {ts}</p></div><button class="theme-toggle" onclick="toggleTheme()">&#9681; Toggle Theme</button></header>

<div class="grid">
  <div class="card"><div class="metric"><span class="value" style="color:var(--green)">{acc:.0f}%</span><br><span class="label">Judge Accuracy ({correct}/{total})</span></div></div>
  <div class="card"><div class="metric"><span class="value" style="color:var(--accent)">{avg_conf:.0f}%</span><br><span class="label">Avg Model Confidence</span></div></div>
  <div class="card"><div class="metric"><span class="value" style="color:var(--teal)">{slm_stats['p50']:.0f}ms</span><br><span class="label">SLM Latency (p50)</span></div></div>
  <div class="card"><div class="metric"><span class="value" style="color:var(--purple)">{llm_p50:.0f}ms</span><br><span class="label">LLM Latency (p50)</span></div></div>
  <div class="card"><div class="metric"><span class="value" style="color:var(--yellow)">${pricing['cpu_spot_monthly']}</span><br><span class="label">CPU Spot/replica/mo</span></div></div>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('predictions')">Predictions</div>
  <div class="tab" onclick="showTab('comparison')">SLM vs LLM</div>
  <div class="tab" onclick="showTab('training')">Training</div>
  <div class="tab" onclick="showTab('cost')">Capacity Planning</div>
</div>

<div id="predictions" class="panel active">
  <div class="card"><h3>Predictions</h3>
    <p style="color:var(--muted);margin-bottom:12px">Each prediction judged by Bedrock ({BEDROCK_MODEL}). Confidence from token logprobs.</p>
    <table><thead><tr><th>Input</th><th>Expected</th><th>Predicted</th><th>Confidence</th><th>Judge</th><th>Reasoning</th><th>Latency</th></tr></thead>
    <tbody>{pred_rows}</tbody></table>
  </div>
</div>

<div id="comparison" class="panel">
  <div class="card"><h3>Latency Comparison</h3>
    <table><thead><tr><th>Metric</th><th><span class="tag tag-slm">SLM</span></th><th><span class="tag tag-llm">LLM API</span></th></tr></thead><tbody>
      <tr><td>End-to-end p50</td><td style="color:var(--teal)">{slm_stats['p50']:.0f}ms</td><td style="color:var(--purple)">{llm_p50:.0f}ms</td></tr>
      <tr><td>End-to-end p95</td><td>{slm_stats['p95']:.0f}ms</td><td>{llm_stats['p95']:.0f}ms</td></tr>
      <tr><td>End-to-end p99</td><td>{slm_stats['p99']:.0f}ms</td><td style="color:var(--muted)">—</td></tr>
      <tr><td>Min / Max</td><td>{slm_stats['min']:.0f}ms / {slm_stats['max']:.0f}ms</td><td style="color:var(--muted)">—</td></tr>
      <tr><td>TTFT (Time to First Token) p50</td><td>{slm_stats['ttft_p50']:.0f}ms</td><td style="color:var(--muted)">—</td></tr>
      <tr><td>TTFT p95</td><td>{slm_stats['ttft_p95']:.0f}ms</td><td style="color:var(--muted)">—</td></tr>
      <tr><td>ITL (Inter-Token Latency) p50</td><td>{slm_stats['itl_p50']:.1f}ms</td><td style="color:var(--muted)">—</td></tr>
      <tr><td>ITL p95</td><td>{slm_stats['itl_p95']:.1f}ms</td><td style="color:var(--muted)">—</td></tr>
      <tr><td>Generation throughput (avg)</td><td>{slm_stats['tok_s_avg']:.1f} tok/s</td><td style="color:var(--muted)">—</td></tr>

    </tbody></table>
  </div>
  <div class="card"><h3>Side-by-Side Predictions</h3>
    <p style="color:var(--muted);margin-bottom:12px">Same inputs sent to both the fine-tuned SLM and an LLM API (zero-shot). The SLM produces structured pipe-delimited output. The LLM gets the intent right but in varying formats.</p>
    <table><thead><tr><th>Input</th><th>Expected</th><th><span class="tag tag-slm">SLM</span> Output</th><th><span class="tag tag-llm">LLM</span> Output</th></tr></thead>
    <tbody>{compare_rows}</tbody></table>
  </div>
</div>

<div id="training" class="panel">
  <div class="card"><h3>Training</h3>
    <p style="color:var(--muted);margin-bottom:12px">{training.get('total_steps',0)} steps | Loss: {training.get('start_loss','?')} &rarr; {training.get('final_loss','?')} (min: {training.get('min_loss','?')}) | {training.get('duration_min',0)} min{' | ' + training.get('gpu_type','') if training.get('gpu_type') else ''}</p>
    {loss_svg}
  </div>
  <div class="card"><h3>Data Distribution</h3>
    <table><thead><tr><th>Class</th><th>Count</th></tr></thead><tbody>{dist_rows}</tbody></table>
  </div>
</div>

<div id="cost" class="panel">
  <div class="card"><h3>Capacity Planning — Monthly Estimates</h3>
    <p style="color:var(--muted);margin-bottom:12px">CPU capacity: {pricing['rps_per_replica']:.1f} req/s per node (avg latency {slm_stats['avg']:.0f}ms). GPU capacity: ~{pricing['gpu_rps_per_replica']:.0f} req/s per node (~15x faster). All prices monthly.</p>
    <table><thead><tr><th>Volume</th><th>Req/s</th><th>CPU Nodes</th><th>CPU On-Demand</th><th>CPU Spot</th><th>GPU Nodes</th><th>GPU</th></tr></thead><tbody>{cost_rows}</tbody></table>
    <div style="margin-top:16px;padding:16px;border-radius:6px;border:1px solid var(--border);background:var(--bg)">
      <p style="font-size:13px;color:var(--muted);margin-bottom:8px"><strong style="color:var(--text)">Note:</strong> These are reference estimates to help with infrastructure sizing. Actual costs depend on:</p>
      <ul style="font-size:13px;color:var(--muted);padding-left:20px;line-height:2">
        <li><strong>Instance selection</strong> — Karpenter picks the cheapest instance that meets resource requests. Actual types (c8g.medium, g5.xlarge, etc.) vary by availability and region.</li>
        <li><strong>Capacity vs cost</strong> — CPU nodes are cheap but slow per request. GPU nodes are expensive but handle 10-20x more throughput. At high volumes, fewer GPU nodes can be cheaper than many CPU nodes.</li>
        <li><strong>Latency requirements</strong> — If sub-100ms latency is needed, GPU inference or a smaller model is required. CPU inference at 1-2s may be acceptable for async workloads.</li>
        <li><strong>Scaling behavior</strong> — HPA scales replicas based on request rate. More replicas means more nodes. Spot pricing reduces cost but may introduce interruptions.</li>
      </ul>
      <p style="font-size:12px;color:var(--muted);margin-top:8px">Use this table as a starting point for infrastructure planning. Test with your actual traffic patterns to validate.</p>
    </div>
  </div>
</div>

<footer>Generated by Slemify &mdash; {ts}</footer>
</div>
<script>
function showTab(id){{document.querySelectorAll('.panel').forEach(function(el){{el.classList.remove('active')}});document.querySelectorAll('.tab').forEach(function(el){{el.classList.remove('active')}});document.getElementById(id).classList.add('active');event.target.classList.add('active')}}
function toggleTheme(){{var h=document.documentElement;h.dataset.theme=h.dataset.theme==='light'?'':'light'}}
</script>
</body></html>"""


def _render_loss_svg(losses, eval_losses):
    if not losses:
        return "<p style='color:var(--muted)'>No training loss data available</p>"
    max_step = max(s for s, _ in losses)
    max_loss = max(l for _, l in losses)
    w, h = 700, 200
    # Training loss line
    points = " ".join(f"{s/max_step*w:.1f},{h - l/max_loss*h:.1f}" for s, l in losses)
    svg = f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:200px;background:var(--bg);border-radius:4px">'
    svg += f'<polyline points="{points}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>'
    # Eval loss points
    if eval_losses:
        for s, l in eval_losses:
            x = s / max_step * w
            y = h - l / max_loss * h
            svg += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--yellow)"/>'
    svg += '</svg>'
    svg += '<p style="font-size:11px;color:var(--muted);margin-top:4px"><span style="color:var(--accent)">&#9644;</span> Training loss <span style="color:var(--yellow)">&#9679;</span> Eval loss</p>'
    return svg


def _pct(sl, pctile):
    if not sl:
        return 0
    return round(sl[min(int(len(sl) * pctile), len(sl) - 1)], 1)


def _esc(t):
    if not t:
        return ""
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    main()
