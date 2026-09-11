#!/usr/bin/env python3
"""Build the held-out retrieval set: N questions, each answerable from exactly
one known chunk of the indexed corpus (the gold chunk).

Run once against a populated index and commit the output
(eval/recall-set.jsonl); recall.py then scores retrieval configurations
against it deterministically, with no judge. Regenerate only when the corpus
changes, and bump the seed so the sample changes with it.

How gold is chosen: sample chunks uniformly at random (seeded) from the
official-docs sources, skip anything too short to carry a fact, and ask
Bedrock to write the question a platform engineer would type that this chunk
answers, in their words, not the chunk's. The chunk id is a hash of its text,
so it is stable across both indexes (same chunker, same corpus) and across
re-indexing.

Usage:
  kubectl port-forward -n slemify svc/opensearch-cluster-master 9200:9200
  python3 eval/make_recall_set.py --n 100 --seed 7
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import boto3
from opensearchpy import OpenSearch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from agent import config  # noqa: E402

OUT = os.path.join(HERE, "recall-set.jsonl")
SOURCES = ["karpenter", "keda", "eks-best-practices"]
MIN_CHARS = 400

PROMPT = """You are writing a realistic question for a retrieval benchmark.

Below is one passage from Kubernetes autoscaling documentation. Write ONE
question that a platform engineer would plausibly type into a support channel,
such that this passage is the best answer to it.

Rules:
- Use the engineer's own words and situation, not the passage's phrasing.
  Do not copy sentences or distinctive phrases from the passage.
- Make it specific enough that this passage, and not a neighboring one,
  answers it. Mention the concrete field, behavior, or scenario the passage
  is about.
- One question, one or two sentences, lowercase is fine, no preamble.

PASSAGE ({source} / {section}):
{text}

QUESTION:"""


def chunk_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def scan_chunks(client: OpenSearch, index: str) -> list[dict]:
    """All chunks from the official-docs sources, via the scroll API."""
    out = []
    resp = client.search(index=index, scroll="2m", size=500, body={
        "query": {"terms": {"source": SOURCES}},
        "_source": ["text", "source", "section"],
    })
    sid = resp["_scroll_id"]
    while resp["hits"]["hits"]:
        for h in resp["hits"]["hits"]:
            s = h["_source"]
            if len(s.get("text", "")) >= MIN_CHARS:
                out.append(s)
        resp = client.scroll(scroll_id=sid, scroll="2m")
        sid = resp["_scroll_id"]
    client.clear_scroll(scroll_id=sid)
    return out


def ask(br, model: str, chunk: dict) -> str:
    resp = br.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": PROMPT.format(
            source=chunk["source"], section=chunk.get("section", ""),
            text=chunk["text"][:3000])}]}],
        inferenceConfig={"maxTokens": 120, "temperature": 0.7},
    )
    return resp["output"]["message"]["content"][0]["text"].strip().strip('"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--index", default=config.INDEX_NAME,
                    help="which index to sample chunks from (same corpus either way)")
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    client = OpenSearch(hosts=[config.OPENSEARCH_URL], use_ssl=False, verify_certs=False)
    chunks = scan_chunks(client, args.index)
    print(f"{len(chunks)} candidate chunks from {SOURCES} (>= {MIN_CHARS} chars)")
    rng = random.Random(args.seed)
    sample = rng.sample(chunks, min(args.n, len(chunks)))

    br = boto3.client("bedrock-runtime")
    rows = []
    for i, c in enumerate(sample, 1):
        q = ask(br, args.model, c)
        rows.append({"id": f"rs-{args.seed}-{i:03d}", "query": q,
                     "gold": {"chunk_id": chunk_id(c["text"]), "source": c["source"],
                              "section": c.get("section", ""), "preview": c["text"][:160]}})
        print(f"  [{i:3}/{len(sample)}] {c['source']:<20} {q[:90]}")
        time.sleep(0.2)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} to {args.out}")


if __name__ == "__main__":
    main()
