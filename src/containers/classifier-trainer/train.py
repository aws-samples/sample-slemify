#!/usr/bin/env python3
"""Slemify encoder-head trainer (CPU, no GPU, no GGUF).

Runs as a K8s Job. Handles three CPU model families that all export the encoder
to ONNX so serving needs neither torch nor sentence-transformers:

  - classification: embed inputs with a frozen encoder, fit a logistic head.
  - scoring:        embed inputs with a frozen encoder, fit a ridge regression head.
  - embedding:      contrastively fine-tune the encoder itself (MultipleNegatives
                    RankingLoss) on (query, positive) pairs, then export it.

Config via environment variables:
  S3_BUCKET, PROJECT, EMBEDDING_MODEL_NAME, HEAD, TASK, EPOCHS
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
# TASK selects classification (label head), scoring (regression head), or
# embedding (contrastive fine-tune of the encoder).
TASK = os.environ.get("TASK", "classification")
# EPOCHS applies to contrastive embedding training (head tasks solve directly).
EPOCHS = int(os.environ.get("EPOCHS", "2") or "2")
if EPOCHS < 1:
    EPOCHS = 2

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


def export_onnx(out_dir, source=None):
    """Export an encoder to ONNX via optimum (transformers library path).

    source defaults to the configured HF model id; for embedding training it
    points at the locally fine-tuned model directory.
    """
    from optimum.exporters.onnx import main_export
    main_export(source or MODEL_NAME, output=out_dir, task="feature-extraction",
                library_name="transformers")


def main():
    log("=== Slemify Classifier Trainer (encoder-head, CPU) ===")
    log(f"Project: {PROJECT}, Encoder: {MODEL_NAME}, Head: {HEAD}, Task: {TASK}")
    if not S3_BUCKET or not PROJECT:
        log("ERROR: S3_BUCKET and PROJECT are required")
        sys.exit(1)

    train_rows = load_jsonl_s3(f"{PROJECT}/processed/train.jsonl")
    eval_rows = load_jsonl_s3(f"{PROJECT}/processed/eval.jsonl")
    log(f"Loaded train={len(train_rows)} eval={len(eval_rows)}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    if TASK == "embedding":
        train_embedding(model, train_rows, eval_rows)
    elif TASK == "scoring":
        train_scoring(model, train_rows, eval_rows)
    else:
        train_classification(model, train_rows, eval_rows)


def _embed(model, texts):
    """Embed a list of texts with CLS pooling + L2 normalize (matches serving)."""
    return np.asarray(
        model.encode(texts, normalize_embeddings=True, batch_size=32,
                     show_progress_bar=False), dtype=np.float32)


def _export_and_upload_encoder(prefix, source=None):
    """Export the encoder to ONNX and upload encoder.onnx + tokenizer.json.

    source defaults to the configured HF model id; embedding training passes the
    locally fine-tuned model directory so the exported graph is the tuned model.
    """
    log("Exporting encoder to ONNX...")
    onnx_dir = "/tmp/onnx-export"
    os.makedirs(onnx_dir, exist_ok=True)
    export_onnx(onnx_dir, source=source)
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
        # Honest baseline: predicting the training mean for every input. The
        # trained head only adds value if its MAE is meaningfully below this.
        baseline_pred = np.full_like(yev, float(ytr.mean()))
        baseline_mae = float(mean_absolute_error(yev, baseline_pred))
        per_query_embed_ms = embed_ms / max(len(Xev_txt), 1)
        metrics.update({"mae": mae, "rmse": rmse, "r2": r2, "correlation": corr,
                        "baseline_mae": baseline_mae,
                        "total": len(yev), "embed_ms_per_query": round(per_query_embed_ms, 1)})
        log(f"\n  MAE: {mae:.4f} (baseline predict-mean MAE: {baseline_mae:.4f})  "
            f"RMSE: {rmse:.4f}  R²: {r2:.3f}  Corr: {corr:.3f} (n={len(yev)})")

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


def _retrieval_metrics(model, eval_pairs, corpus_texts, ks=(1, 5, 10)):
    """Compute recall@k and MRR for query->positive retrieval over the corpus.

    Each eval pair's positive document is located in the corpus; we embed all
    corpus docs once and each query, then rank by cosine similarity.
    """
    if not eval_pairs or not corpus_texts:
        return {}
    # Map positive text -> its index in the corpus (exact match on chunk text).
    corpus_index = {t: i for i, t in enumerate(corpus_texts)}
    queries, gold = [], []
    for p in eval_pairs:
        pos = p.get("positive", "")
        if pos in corpus_index:
            queries.append(p.get("query", ""))
            gold.append(corpus_index[pos])
    if not queries:
        return {}

    doc_vecs = _embed(model, corpus_texts)
    q_vecs = _embed(model, queries)
    # Cosine similarity (vectors are already L2-normalized) -> [n_queries, n_docs].
    sims = q_vecs @ doc_vecs.T

    max_k = max(ks)
    # Top-k doc indices per query, best first.
    topk = np.argsort(-sims, axis=1)[:, :max_k]
    metrics = {}
    for k in ks:
        hits = sum(1 for i, g in enumerate(gold) if g in topk[i, :k])
        metrics[f"recall@{k}"] = hits / len(gold)
    # MRR over the top max_k.
    rr = 0.0
    for i, g in enumerate(gold):
        row = topk[i].tolist()
        if g in row:
            rr += 1.0 / (row.index(g) + 1)
    metrics["mrr"] = rr / len(gold)
    metrics["eval_queries"] = len(gold)
    return metrics


def train_embedding(model, train_rows, eval_rows):
    """Contrastively fine-tune the encoder on (query, positive) pairs."""
    from sentence_transformers import InputExample, losses
    from torch.utils.data import DataLoader

    # Cap sequence length to keep CPU activation memory bounded. Retrieval
    # chunks and queries are short; 256 tokens is ample and ~4x lighter than the
    # 512 default for the contrastive forward/backward pass.
    model.max_seq_length = min(getattr(model, "max_seq_length", 256) or 256, 256)

    pairs = [(r.get("query", ""), r.get("positive", ""))
             for r in train_rows if r.get("query") and r.get("positive")]
    if not pairs:
        log("ERROR: no (query, positive) training pairs")
        sys.exit(1)
    log(f"Training pairs: {len(pairs)} (max_seq_length={model.max_seq_length})")

    # Build the retrieval corpus for eval: prefer corpus.jsonl, fall back to the
    # set of positive documents seen in train+eval.
    try:
        corpus_rows = load_jsonl_s3(f"{PROJECT}/processed/corpus.jsonl")
        corpus_texts = [c.get("text", "") for c in corpus_rows if c.get("text")]
    except Exception:
        corpus_texts = []
    if not corpus_texts:
        seen = {r.get("positive", "") for r in train_rows + eval_rows if r.get("positive")}
        corpus_texts = sorted(t for t in seen if t)
    # Eval pairs are generated from an independent source, so their gold
    # documents may not be in the training corpus. Add them so each eval query
    # has its positive present; the training chunks act as distractors. This
    # keeps recall@k honest (gold among a realistic candidate set).
    eval_positives = {r.get("positive", "") for r in eval_rows if r.get("positive")}
    corpus_set = set(corpus_texts)
    corpus_texts = corpus_texts + [t for t in eval_positives if t and t not in corpus_set]
    log(f"Retrieval corpus: {len(corpus_texts)} documents")

    # Baseline retrieval metrics (stock encoder) for a before/after comparison.
    baseline = _retrieval_metrics(model, eval_rows, corpus_texts)
    if baseline:
        log(f"Baseline (stock encoder): recall@5={baseline.get('recall@5', 0):.3f} "
            f"mrr={baseline.get('mrr', 0):.3f}")

    # Contrastive fine-tune with in-batch negatives (MultipleNegativesRankingLoss).
    examples = [InputExample(texts=[q, p]) for q, p in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=16, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = max(1, int(len(loader) * EPOCHS * 0.1))
    log(f"Fine-tuning: {EPOCHS} epoch(s), {len(loader)} batches/epoch, warmup={warmup}")
    # sentence-transformers 3.x routes fit() through the HF Trainer, which writes
    # a relative "checkpoints/" dir. The container root filesystem is read-only;
    # only /tmp is writable, so run from there.
    os.makedirs("/tmp/st-train", exist_ok=True)
    os.chdir("/tmp/st-train")
    t0 = time.time()
    model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS,
              warmup_steps=warmup, show_progress_bar=False)
    train_s = time.time() - t0
    log(f"Fine-tune complete in {train_s:.0f}s")

    tuned = _retrieval_metrics(model, eval_rows, corpus_texts)
    if tuned:
        log(f"Tuned encoder: recall@5={tuned.get('recall@5', 0):.3f} "
            f"mrr={tuned.get('mrr', 0):.3f}")

    dim = int(model.get_sentence_embedding_dimension())
    metrics = {"task": "embedding", "embedding_dim": dim,
               "train_samples": len(pairs), "eval_queries": tuned.get("eval_queries", 0),
               "corpus_size": len(corpus_texts), "epochs": EPOCHS,
               "train_seconds": round(train_s, 1),
               "baseline": baseline, "tuned": tuned}

    # Persist the fine-tuned model, then export THAT to ONNX.
    tuned_dir = "/tmp/tuned-model"
    model.save(tuned_dir)

    prefix = f"models/{PROJECT}"
    log("Uploading metrics to S3...")
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}/metrics.json",
                  Body=json.dumps(metrics, indent=2).encode(), ContentType="application/json")

    _export_and_upload_encoder(prefix, source=tuned_dir)
    log(f"Artifacts uploaded to s3://{S3_BUCKET}/{prefix}/")
    log("=== Training complete ===")



if __name__ == "__main__":
    main()
