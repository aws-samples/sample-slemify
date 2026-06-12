#!/usr/bin/env python3
"""Slemify encoder-head classifier training.

Frozen encoder + trained classifier head (CPU only, no GPU, no GGUF).

Pipeline:
  1. Read train.jsonl / eval.jsonl from S3 (instruction/input/output records).
  2. Embed inputs via the in-cluster encoder service (TEI-compatible /embed).
  3. Fit a classifier head (logistic regression) on embeddings -> labels.
  4. Evaluate on the eval set (exact-match accuracy + per-class metrics).
  5. Upload head.json, labels.json, metrics.json to S3.

Config via environment variables:
  S3_BUCKET, PROJECT, ENCODER_URL, HEAD ("logistic"|"linear"|"mlp")
"""
import json
import os
import sys
import time
import urllib.request

import boto3
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

S3_BUCKET = os.environ.get("S3_BUCKET", "")
PROJECT = os.environ.get("PROJECT", "")
ENCODER_URL = os.environ.get("ENCODER_URL", "http://localhost:8080")
HEAD = os.environ.get("HEAD", "logistic")
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "32"))

s3 = boto3.client("s3")


def log(msg):
    print(msg, flush=True)


def load_jsonl_s3(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    rows = []
    for line in obj["Body"].read().decode().strip().split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def to_xy(rows):
    """Extract input text and the single classification label.

    Records use pipe-delimited output for label tasks; for single-dimension
    classification the label is the last pipe field (or the whole string).
    """
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


def embed_batch(texts):
    body = json.dumps({"inputs": texts}).encode()
    req = urllib.request.Request(
        f"{ENCODER_URL}/embed", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def embed_all(texts, label=""):
    out = []
    t0 = time.time()
    for i in range(0, len(texts), EMBED_BATCH):
        out.extend(embed_batch(texts[i:i + EMBED_BATCH]))
        log(f"  embed {label}: {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")
    dt = (time.time() - t0) * 1000
    return np.array(out, dtype=np.float32), dt


def build_head():
    # Phase 2a supports the logistic head. linear/mlp are accepted by the
    # schema and fall back to logistic until implemented.
    return LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")


def main():
    log("=== Slemify Classifier Training (encoder-head, CPU) ===")
    log(f"Project: {PROJECT}, Encoder: {ENCODER_URL}, Head: {HEAD}")

    if not S3_BUCKET or not PROJECT:
        log("ERROR: S3_BUCKET and PROJECT are required")
        sys.exit(1)

    train_rows = load_jsonl_s3(f"{PROJECT}/processed/train.jsonl")
    eval_rows = load_jsonl_s3(f"{PROJECT}/processed/eval.jsonl")
    log(f"Loaded train={len(train_rows)} eval={len(eval_rows)}")

    Xtr_txt, ytr = to_xy(train_rows)
    Xev_txt, yev = to_xy(eval_rows)
    if not Xtr_txt:
        log("ERROR: no training records")
        sys.exit(1)

    classes = sorted(set(ytr))
    log(f"Classes ({len(classes)}): {classes}")

    log("Embedding train set...")
    Xtr, _ = embed_all(Xtr_txt, "train")
    if Xev_txt:
        log("Embedding eval set...")
        Xev, embed_ms = embed_all(Xev_txt, "eval")
    else:
        Xev, embed_ms = np.empty((0, Xtr.shape[1]), dtype=np.float32), 0.0

    dim = int(Xtr.shape[1])
    log(f"Embedding dim: {dim}")

    log("Fitting classifier head...")
    clf = build_head()
    clf.fit(Xtr, ytr)

    metrics = {"embedding_dim": dim, "head": HEAD, "num_classes": len(classes),
               "train_samples": len(ytr), "eval_samples": len(yev)}

    if Xev_txt:
        pred = clf.predict(Xev)
        acc = float(accuracy_score(yev, pred))
        per_query_embed_ms = embed_ms / max(len(Xev_txt), 1)
        p, r, f1, _ = precision_recall_fscore_support(
            yev, pred, labels=clf.classes_, average=None, zero_division=0)
        per_class = {}
        for i, c in enumerate(clf.classes_):
            per_class[c] = {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f1[i])}
        metrics.update({
            "accuracy": acc,
            "correct": int(round(acc * len(yev))),
            "total": len(yev),
            "per_class": per_class,
            "embed_ms_per_query": round(per_query_embed_ms, 1),
        })
        log(f"\n  Accuracy (exact match): {acc * 100:.1f}% ({metrics['correct']}/{len(yev)})")
        log(f"  Embed latency/query: ~{per_query_embed_ms:.0f}ms")

    # Serialize the head: classes + coefficients + intercept (logistic).
    head_obj = {
        "head": HEAD,
        "embedding_dim": dim,
        "classes": list(clf.classes_),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
    }

    log("Uploading artifacts to S3...")
    prefix = f"models/{PROJECT}"
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/head.json",
                  Body=json.dumps(head_obj).encode(), ContentType="application/json")
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/labels.json",
                  Body=json.dumps({"classes": list(clf.classes_)}).encode(),
                  ContentType="application/json")
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/metrics.json",
                  Body=json.dumps(metrics, indent=2).encode(),
                  ContentType="application/json")
    log(f"Artifacts uploaded to s3://{S3_BUCKET}/{prefix}/")
    log("=== Training complete ===")


if __name__ == "__main__":
    main()
