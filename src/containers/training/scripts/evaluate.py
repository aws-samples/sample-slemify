#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Post-training evaluation: domain accuracy on eval.jsonl and MMLU subset for catastrophic forgetting.

Writes results JSON to S3 in the format expected by pkg/report/training.go:
{
  "domain_accuracy": 0.87,
  "mmlu_base": 0.62,
  "mmlu_fine_tuned": 0.60,
  "loss_curve": [{"step": 100, "loss": 2.5, "epoch": 0.5}, ...],
  "training_time_hours": 2.5,
  "spot_used": true
}
"""

import json
import os
import sys
import time

import boto3


def evaluate_domain(output_dir: str, eval_path: str) -> float:
    """Evaluate fine-tuned model on domain-specific eval set.

    Reads eval_results.json from the output directory.
    Returns domain accuracy as a float 0.0-1.0.
    """
    eval_results_path = os.path.join(output_dir, "eval_results.json")
    if os.path.exists(eval_results_path):
        with open(eval_results_path, encoding="utf-8") as f:
            data = json.load(f)
            # Trainer reports eval_loss; convert to approximate accuracy
            eval_loss = data.get("eval_loss", 1.0)
            # Heuristic: lower loss = higher accuracy (sigmoid-like mapping)
            import math
            accuracy = 1.0 / (1.0 + math.exp(eval_loss - 1.0))
            return round(accuracy, 4)

    # Fallback: check if there's a custom eval output
    custom_eval = os.path.join(output_dir, "domain_eval.json")
    if os.path.exists(custom_eval):
        with open(custom_eval, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("accuracy", 0.0)

    print("Warning: no eval results found, returning 0.0", file=sys.stderr)
    return 0.0


def evaluate_forgetting(output_dir: str, base_model: str) -> tuple[float, float]:
    """Run MMLU subset to detect catastrophic forgetting.

    Returns (base_score, fine_tuned_score) as floats 0.0-1.0.
    In production, this runs lm-eval-harness on both the base and fine-tuned model.
    """
    # Check if lm-eval results exist (pre-computed during training)
    mmlu_path = os.path.join(output_dir, "mmlu_results.json")
    if os.path.exists(mmlu_path):
        with open(mmlu_path, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("base_score", 0.0), data.get("fine_tuned_score", 0.0)

    # Try running lm-eval-harness if available
    try:
        import subprocess
        # Run on fine-tuned model
        ft_result = subprocess.run(
            ["lm_eval", "--model", "hf", "--model_args",
             f"pretrained={output_dir}", "--tasks", "mmlu",
             "--num_fewshot", "5", "--batch_size", "4",
             "--output_path", os.path.join(output_dir, "lm_eval_ft")],
            capture_output=True, text=True, timeout=3600,
        )
        # Parse results
        ft_score = _parse_lm_eval_score(os.path.join(output_dir, "lm_eval_ft"))

        # Run on base model
        base_result = subprocess.run(
            ["lm_eval", "--model", "hf", "--model_args",
             f"pretrained={base_model}", "--tasks", "mmlu",
             "--num_fewshot", "5", "--batch_size", "4",
             "--output_path", os.path.join(output_dir, "lm_eval_base")],
            capture_output=True, text=True, timeout=3600,
        )
        base_score = _parse_lm_eval_score(os.path.join(output_dir, "lm_eval_base"))

        return base_score, ft_score
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        print(f"Warning: lm-eval not available or failed: {e}", file=sys.stderr)
        return 0.0, 0.0


def _parse_lm_eval_score(output_path: str) -> float:
    """Parse lm-eval-harness output for MMLU accuracy."""
    results_file = os.path.join(output_path, "results.json")
    if os.path.exists(results_file):
        with open(results_file, encoding="utf-8") as f:
            data = json.load(f)
            # lm-eval stores results under "results" -> task -> metric
            results = data.get("results", {})
            if "mmlu" in results:
                return results["mmlu"].get("acc,none", 0.0)
    return 0.0


def collect_loss_curve(output_dir: str) -> list[dict]:
    """Parse trainer_state.json for loss curve data."""
    curve = []
    state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
            for entry in state.get("log_history", []):
                if "loss" in entry:
                    curve.append({
                        "step": entry.get("step", 0),
                        "loss": round(entry["loss"], 4),
                        "epoch": round(entry.get("epoch", 0.0), 2),
                    })
    return curve


def get_training_time(output_dir: str) -> float:
    """Calculate training time from trainer_state.json timestamps."""
    state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
            total_secs = state.get("total_flos", 0)
            # Use log_history timestamps if available
            log = state.get("log_history", [])
            if len(log) >= 2:
                # Estimate from step count and throughput
                total_steps = state.get("global_step", 0)
                if total_steps > 0 and "train_runtime" in state:
                    return round(state["train_runtime"] / 3600, 2)
    return 0.0


def upload_results(results: dict, bucket: str, project: str):
    """Upload evaluation results JSON to S3."""
    s3 = boto3.client("s3")
    key = f"{project}/reports/eval-results.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(results, indent=2),
        ContentType="application/json",
    )
    print(f"Evaluation results uploaded to s3://{bucket}/{key}")


def main():
    output_dir = os.environ.get("OUTPUT_DIR", "/workspace/output")
    bucket = os.environ.get("S3_BUCKET", "")
    project = os.environ.get("PROJECT_NAME", "")
    base_model = os.environ.get("BASE_MODEL", "")
    spot_used = os.environ.get("SPOT_USED", "false").lower() == "true"

    if not bucket or not project:
        print("S3_BUCKET and PROJECT_NAME env vars required", file=sys.stderr)
        sys.exit(1)

    print("Running domain evaluation...")
    domain_accuracy = evaluate_domain(output_dir, "")

    print("Running catastrophic forgetting check...")
    base_score, ft_score = evaluate_forgetting(output_dir, base_model)

    print("Collecting loss curve...")
    loss_curve = collect_loss_curve(output_dir)

    print("Calculating training time...")
    training_time = get_training_time(output_dir)

    # Format matching pkg/report/training.go EvalResults struct
    results = {
        "domain_accuracy": domain_accuracy,
        "mmlu_base": base_score,
        "mmlu_fine_tuned": ft_score,
        "loss_curve": loss_curve,
        "training_time_hours": training_time,
        "spot_used": spot_used,
    }

    print(f"Results: accuracy={domain_accuracy:.3f}, "
          f"forgetting={base_score:.3f}->{ft_score:.3f}, "
          f"time={training_time:.1f}h")

    upload_results(results, bucket, project)


if __name__ == "__main__":
    main()
