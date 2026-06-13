#!/usr/bin/env python3
"""Slemify encoder-head classifier trainer (CPU, no GPU, no GGUF).

Runs as a K8s Job. Embeds training inputs in-process with sentence-transformers,
fits a classifier head, and exports the encoder to ONNX so serving needs neither
torch nor sentence-transformers.

Steps:
  1. Read train.jsonl / eval.jsonl from S3.
  2. Embed inputs in-process (sentence-transformers, CLS pooling, normalized).
  3. Fit a logistic head on embeddings -> labels; evaluate on eval.
  4. Export the encoder to ONNX (optimum) + tokenizer.
  5. Upload head.json, labels.json, metrics.json, encoder.onnx, tokenizer.json.

Config via environment variables:
  S3_BUCKET, PROJECT, EMBEDDING_MODEL_NAME, HEAD
"""
import json
import os
import sys
import time

import boto3
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support,
                             r2_score)

S3_BUCKET = os.environ.get("S3_BUCKET", "")
PROJECT = os.environ.get("PROJECT", "")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
HEAD = os.environ.get("HEAD", "logistic")
# TASK selects classification (label head) vs scoring (regression head).
TASK = os.environ.get("TASK", "classification")

s3 = boto3.client("s3")


def log(msg):
    print(msg, flush=True)


def load_jsonl_s3(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return [json.loads(l) for l in obj["Body"].read().decode().strip().split("\n") if l.strip()]


def to_xy(rows):
    """Extract input text and the single classification label."""
    X, y = [], []
    for r in rows:
        text = r.get("input", "")
        instruction = r.get("instruction", "")
        if instruction:
            text = f"{instruction}\n\n{text}"
        out = r.get("output", "").strip()
        label = out.split("|")[-1].strip() if "|" in out else out
        if text and label:
            X.append(text)
            y.append(label)
    return X, y


def to_xy_scoring(rows):
    """Extract input text and the numeric target for regression/scoring."""
    X, y = [], []
    for r in rows:
        text = r.get("input", "")
        instruction = r.get("instruction", "")
        if instruction:
            text = f"{instruction}\n\n{text}"
        out = r.get("output", "").strip()
        try:
            val = float(out)
        except ValueError:
            continue
        if text:
            X.append(text)
            y.append(val)
    return X, np.asarray(y, dtype=np.float32)


def export_onnx(out_dir):
    """Export the encoder to ONNX via optimum (transformers library path)."""
    from optimum.exporters.onnx import main_export
    main_export(MODEL_NAME, output=out_dir, task="feature-extraction",
                library_name="transformers")


def main():
    log("=== Slemify Classifier Trainer (encoder-head, CPU) ===")
    log(f"Project: {PROJECT}, Encoder: {MODEL_NAME}, Head: {HEAD}, Task: {TASK}")
    if not S3_BUCKET or not PROJECT:
        log("ERROR: S3_BUCKET and PROJECT are required")
        sys.exit(1)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    train_rows = load_jsonl_s3(f"{PROJECT}/processed/train.jsonl")
    eval_rows = load_jsonl_s3(f"{PROJECT}/processed/eval.jsonl")
    log(f"Loaded train={len(train_rows)} eval={len(eval_rows)}")

    is_scoring = (TASK == "scoring")
    if is_scoring:
        train_scoring(model, train_rows, eval_rows)
    else:
        train_classification(model, train_rows, eval_rows)


def _embed(model, texts):
    """Embed a list of texts with CLS pooling + L2 normalize (matches serving)."""
    return np.asarray(
        model.encode(texts, normalize_embeddings=True, batch_size=32,
                     show_progress_bar=False), dtype=np.float32)


def _export_and_upload_encoder(prefix):
    """Export the frozen encoder to ONNX and upload encoder.onnx + tokenizer.json."""
    log("Exporting encoder to ONNX...")
    onnx_dir = "/tmp/onnx-export"
    os.makedirs(onnx_dir, exist_ok=True)
    export_onnx(onnx_dir)
    for fname in ("model.onnx", "tokenizer.json"):
        fpath = os.path.join(onnx_dir, fname)
        key = f"{prefix}/encoder.onnx" if fname == "model.onnx" else f"{prefix}/tokenizer.json"
        s3.upload_file(fpath, S3_BUCKET, key)
        log(f"  uploaded {key} ({os.path.getsize(fpath) / 1048576:.0f} MB)")


def train_classification(model, train_rows, eval_rows):
    Xtr_txt, ytr = to_xy(train_rows)
    Xev_txt, yev = to_xy(eval_rows)
    if not Xtr_txt:
        log("ERROR: no training records")
        sys.exit(1)

    classes = sorted(set(ytr))
    log(f"Classes ({len(classes)}): {classes}")

    log("Embedding train set (in-process)...")
    Xtr = _embed(model, Xtr_txt)
    t0 = time.time()
    if Xev_txt:
        log("Embedding eval set...")
        Xev = _embed(model, Xev_txt)
        embed_ms = (time.time() - t0) * 1000
    else:
        Xev, embed_ms = np.empty((0, Xtr.shape[1]), dtype=np.float32), 0.0

    dim = int(Xtr.shape[1])
    log(f"Embedding dim: {dim}")

    log("Fitting classifier head...")
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    clf.fit(Xtr, ytr)

    metrics = {"task": "classification", "embedding_dim": dim, "head": HEAD,
               "num_classes": len(classes), "train_samples": len(ytr),
               "eval_samples": len(yev)}

    if Xev_txt:
        pred = clf.predict(Xev)
        acc = float(accuracy_score(yev, pred))
        per_query_embed_ms = embed_ms / max(len(Xev_txt), 1)
        p, r, f1, _ = precision_recall_fscore_support(
            yev, pred, labels=clf.classes_, average=None, zero_division=0)
        per_class = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
                     for i, c in enumerate(clf.classes_)}
        metrics.update({"accuracy": acc, "correct": int(round(acc * len(yev))),
                        "total": len(yev), "per_class": per_class,
                        "embed_ms_per_query": round(per_query_embed_ms, 1)})
        log(f"\n  Accuracy (exact match): {acc * 100:.1f}% ({metrics['correct']}/{len(yev)})")

    head_obj = {"type": "classification", "head": HEAD, "embedding_dim": dim,
                "pooling": "cls", "classes": list(clf.classes_),
                "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}

    prefix = f"models/{PROJECT}"
    log("Uploading head/labels/metrics to S3...")
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/head.json",
                  Body=json.dumps(head_obj).encode(), ContentType="application/json")
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/labels.json",
                  Body=json.dumps({"classes": list(clf.classes_)}).encode(),
                  ContentType="application/json")
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/metrics.json",
                  Body=json.dumps(metrics, indent=2).encode(), ContentType="application/json")

    _export_and_upload_encoder(prefix)
    log(f"Artifacts uploaded to s3://{S3_BUCKET}/{prefix}/")
    log("=== Training complete ===")


def train_scoring(model, train_rows, eval_rows):
    Xtr_txt, ytr = to_xy_scoring(train_rows)
    Xev_txt, yev = to_xy_scoring(eval_rows)
    if not Xtr_txt:
        log("ERROR: no training records")
        sys.exit(1)

    log(f"Scoring targets: train={len(ytr)} (mean={float(ytr.mean()):.3f}), "
        f"eval={len(yev)}")

    log("Embedding train set (in-process)...")
    Xtr = _embed(model, Xtr_txt)
    t0 = time.time()
    if Xev_txt:
        log("Embedding eval set...")
        Xev = _embed(model, Xev_txt)
        embed_ms = (time.time() - t0) * 1000
    else:
        Xev, embed_ms = np.empty((0, Xtr.shape[1]), dtype=np.float32), 0.0

    dim = int(Xtr.shape[1])
    log(f"Embedding dim: {dim}")

    log("Fitting regression head (Ridge)...")
    reg = Ridge(alpha=1.0)
    reg.fit(Xtr, ytr)

    metrics = {"task": "scoring", "embedding_dim": dim, "head": HEAD,
               "train_samples": len(ytr), "eval_samples": len(yev)}

    if Xev_txt:
        pred = np.clip(reg.predict(Xev), 0.0, 1.0)
        mae = float(mean_absolute_error(yev, pred))
        rmse = float(np.sqrt(mean_squared_error(yev, pred)))
        r2 = float(r2_score(yev, pred)) if len(yev) > 1 else 0.0
        # Pearson correlation between predicted and true scores.
        if len(yev) > 1 and np.std(pred) > 1e-9 and np.std(yev) > 1e-9:
            corr = float(np.corrcoef(pred, yev)[0, 1])
        else:
            corr = 0.0
        per_query_embed_ms = embed_ms / max(len(Xev_txt), 1)
        metrics.update({"mae": mae, "rmse": rmse, "r2": r2, "correlation": corr,
                        "total": len(yev), "embed_ms_per_query": round(per_query_embed_ms, 1)})
        log(f"\n  MAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.3f}  Corr: {corr:.3f} "
            f"(n={len(yev)})")

    # Regression head: single weight vector + scalar intercept.
    head_obj = {"type": "regression", "head": HEAD, "embedding_dim": dim,
                "pooling": "cls", "coef": reg.coef_.tolist(),
                "intercept": float(reg.intercept_),
                "score_min": 0.0, "score_max": 1.0}

    prefix = f"models/{PROJECT}"
    log("Uploading head/metrics to S3...")
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/head.json",
                  Body=json.dumps(head_obj).encode(), ContentType="application/json")
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/metrics.json",
                  Body=json.dumps(metrics, indent=2).encode(), ContentType="application/json")

    _export_and_upload_encoder(prefix)
    log(f"Artifacts uploaded to s3://{S3_BUCKET}/{prefix}/")
    log("=== Training complete ===")


if __name__ == "__main__":
    main()
