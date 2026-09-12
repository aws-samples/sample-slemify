"""Build the Bedrock (Titan) index from the chunks already in the tuned-encoder index.

index-knowledge.py clones the documentation repos, chunks them, embeds, and
indexes. Once that has run for the tuned encoder, the chunks are already in
OpenSearch; the Titan index for the EMBED=bedrock seat only needs the same
chunks embedded again with Titan. This script does that without cloning or
chunking anything, so the two indexes hold exactly the same corpus.

Usage (port-forwarded, or in-cluster with OPENSEARCH_URL set):
  python3 index-bedrock-from-index.py
Env: OPENSEARCH_URL, INDEX_NAME (source), BEDROCK_INDEX_NAME (target),
     BEDROCK_EMBED_MODEL, BEDROCK_EMBED_DIM, AWS_REGION.
Requires: boto3, opensearch-py (both in the orchestrator image).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from opensearchpy import OpenSearch, helpers

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
SOURCE_INDEX = os.environ.get("INDEX_NAME", "k8s-autoscaling-knowledge")
TARGET_INDEX = os.environ.get("BEDROCK_INDEX_NAME", "k8s-autoscaling-knowledge-bedrock")
MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
DIM = int(os.environ.get("BEDROCK_EMBED_DIM", "1024"))
WORKERS = int(os.environ.get("WORKERS", "8"))

bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "eu-west-1"))


def embed(text: str) -> list[float]:
    for attempt in range(6):
        try:
            resp = bedrock.invoke_model(
                modelId=MODEL,
                body=json.dumps({"inputText": text[:8000], "dimensions": DIM, "normalize": True}),
            )
            return json.loads(resp["body"].read())["embedding"]
        except Exception as e:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError("unreachable")


def main():
    host = OPENSEARCH_URL.replace("http://", "").replace("https://", "")
    client = OpenSearch(hosts=[host], timeout=60)
    total = client.count(index=SOURCE_INDEX)["count"]
    print(f"Source {SOURCE_INDEX}: {total} chunks -> {TARGET_INDEX} via {MODEL} ({DIM}d)", flush=True)

    if client.indices.exists(index=TARGET_INDEX):
        print(f"  Deleting existing {TARGET_INDEX}", flush=True)
        client.indices.delete(index=TARGET_INDEX)
    client.indices.create(index=TARGET_INDEX, body={
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {
            "text": {"type": "text"},
            "source": {"type": "keyword"},
            "section": {"type": "keyword"},
            "embedding": {"type": "knn_vector", "dimension": DIM,
                          "method": {"name": "hnsw", "space_type": "innerproduct", "engine": "faiss"}},
        }},
    })

    done = 0
    batch = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for hit in helpers.scan(client, index=SOURCE_INDEX, query={"query": {"match_all": {}}},
                                _source=["text", "source", "section"], size=200):
            batch.append(hit)
            if len(batch) < 100:
                continue
            vectors = list(pool.map(lambda h: embed(h["_source"]["text"]), batch))
            helpers.bulk(client, ({"_index": TARGET_INDEX, "_id": h["_id"],
                                   "_source": {**h["_source"], "embedding": v}}
                                  for h, v in zip(batch, vectors)))
            done += len(batch)
            batch = []
            print(f"  {done}/{total}", flush=True)
        if batch:
            vectors = list(pool.map(lambda h: embed(h["_source"]["text"]), batch))
            helpers.bulk(client, ({"_index": TARGET_INDEX, "_id": h["_id"],
                                   "_source": {**h["_source"], "embedding": v}}
                                  for h, v in zip(batch, vectors)))
            done += len(batch)
    client.indices.refresh(index=TARGET_INDEX)
    print(f"Done: {client.count(index=TARGET_INDEX)['count']} chunks in {TARGET_INDEX}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
