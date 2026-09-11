#!/usr/bin/env python3
"""Retrieval quality for the EMBED and RERANK seats: recall@2, recall@5, MRR
against the held-out set (eval/recall-set.jsonl). Deterministic, no judge.

Three configurations, run in one go or one at a time:

  bedrock          Titan embeddings, the Bedrock index, vector order (monolith)
  slemify          the tuned encoder, the Slemify index, vector order
  slemify+rerank   the tuned encoder, then the cross-encoder re-orders

Each uses the same hybrid candidate pool the agent uses (dense k-NN plus BM25,
same sizes), so the numbers describe what the agent would hand the analyst.
Gold is a chunk id (hash of the chunk text), stable across both indexes.

Usage (port-forward what the mode needs):
  kubectl port-forward -n slemify svc/opensearch-cluster-master 9200:9200
  kubectl port-forward -n slemify svc/k8s-autoscaling-retriever-inference 8083:8080
  kubectl port-forward -n slemify svc/k8s-autoscaling-reranker 8084:80
  python3 eval/recall.py                         # all three
  python3 eval/recall.py --mode slemify+rerank   # one
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

# The reranker helper checks config.RERANK; force it on here so this script can
# exercise the reranker regardless of how the orchestrator is configured.
os.environ.setdefault("RERANK", "on")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from agent import config, retrieval  # noqa: E402

SET = os.path.join(HERE, "recall-set.jsonl")
RESULTS_DIR = os.path.join(HERE, "results")
MODES = ["bedrock", "slemify", "slemify+rerank"]
RANK_DEPTH = 10  # MRR counts a hit anywhere in the top 10


def chunk_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def _search(index: str, body: dict) -> list[dict]:
    res = config.opensearch.search(index=index, body={**body, "_source": ["text", "source", "section"]})
    return [h["_source"] for h in res["hits"]["hits"]]


def candidates(mode: str, query: str) -> list[dict]:
    """Hybrid pool exactly as the agent builds it: dense first, then BM25, de-duplicated."""
    if mode == "bedrock":
        emb, index = retrieval._embed_bedrock(query), config.BEDROCK_INDEX_NAME
    else:
        emb, index = retrieval._embed_slemify(query), config.INDEX_NAME
    dense = _search(index, {"size": config.RETRIEVE_CANDIDATES,
                            "query": {"knn": {"embedding": {"vector": emb, "k": config.RETRIEVE_CANDIDATES}}}})
    lex = _search(index, {"size": config.LEXICAL_CANDIDATES, "query": {"match": {"text": query}}})
    seen, out = set(), []
    for s in dense + lex:
        cid = chunk_id(s["text"])
        if cid not in seen:
            seen.add(cid)
            out.append({**s, "chunk_id": cid})
    return out


def rank(mode: str, query: str, pool: list[dict]) -> list[str]:
    """Ranked chunk ids, top RANK_DEPTH."""
    if mode != "slemify+rerank":
        return [c["chunk_id"] for c in pool[:RANK_DEPTH]]
    docs = [f"[{c['source']} / {c.get('section', '')}]\n{c['text']}" for c in pool]
    by_doc = {d: c["chunk_id"] for d, c in zip(docs, pool)}
    ordered = retrieval.rerank_docs(query[:config.RERANK_QUERY_CHARS], docs, RANK_DEPTH)
    return [by_doc[d] for d in ordered]


def score(mode: str, items: list[dict]) -> dict:
    hits2 = hits5 = 0
    rr, ms = [], []
    misses = []
    for it in items:
        t0 = time.perf_counter()
        pool = candidates(mode, it["query"])
        ranked = rank(mode, it["query"], pool)
        ms.append(round((time.perf_counter() - t0) * 1000))
        gold = it["gold"]["chunk_id"]
        pos = ranked.index(gold) + 1 if gold in ranked else None
        if pos and pos <= 2:
            hits2 += 1
        if pos and pos <= 5:
            hits5 += 1
        rr.append(1 / pos if pos else 0.0)
        if not pos:
            misses.append(it["id"])
    n = len(items)
    return {"mode": mode, "n": n, "recall@2": round(hits2 / n, 3), "recall@5": round(hits5 / n, 3),
            "mrr": round(statistics.fmean(rr), 3), "latency_p50_ms": statistics.median(ms),
            "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES + ["all"], default="all")
    ap.add_argument("--set", default=SET)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N (smoke)")
    args = ap.parse_args()

    with open(args.set, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        items = items[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]
    print(f"{len(items)} held-out queries\n")
    print(f"  {'mode':<16} {'recall@2':>9} {'recall@5':>9} {'MRR':>7} {'p50 ms':>8}")
    results = []
    for mode in modes:
        r = score(mode, items)
        results.append(r)
        print(f"  {mode:<16} {r['recall@2']:>9.3f} {r['recall@5']:>9.3f} {r['mrr']:>7.3f} {r['latency_p50_ms']:>8.0f}")
    if len(results) > 1:
        base = results[0]
        for r in results[1:]:
            d2 = r["recall@2"] - base["recall@2"]
            print(f"\n  {r['mode']} vs {base['mode']}: recall@2 {d2:+.3f}, "
                  f"MRR {r['mrr'] - base['mrr']:+.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(RESULTS_DIR, f"recall-{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"set": os.path.basename(args.set), "results": results}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
