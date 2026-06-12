"""Index Karpenter and KEDA documentation into OpenSearch for RAG.

Clones official doc repos, chunks markdown files, generates embeddings
via the in-cluster TEI embedding pod (bge-base-en-v1.5), and indexes into
OpenSearch k-NN.

Usage:
  kubectl port-forward -n slemify svc/opensearch-cluster-master 9200:9200
  kubectl port-forward -n slemify svc/k8s-autoscaling-embedding 8083:80
  python3 index-knowledge.py                    # Full reindex
  python3 index-knowledge.py --append --source=aws-blog  # Add blogs only

Requires: pip install opensearch-py httpx gitpython requests beautifulsoup4
"""

import os
import re
import shutil
import sys
import tempfile

import httpx
from git import Repo
from opensearchpy import OpenSearch

# --- Configuration ---

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "k8s-autoscaling-knowledge")
# In-cluster embedding model (TEI serving bge-base-en-v1.5, 768 dimensions).
# Must match the dimension in the index mapping and in server.py at query time.
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:8083")
EMBEDDING_DIM = 768
CHUNK_SIZE = 2000  # chars (~500 tokens)
CHUNK_OVERLAP = 200

SOURCES = [
    {
        "name": "karpenter",
        "repo": "https://github.com/aws/karpenter-provider-aws.git",
        "branch": "main",
        "path": "website/content/en/docs",
    },
    {
        "name": "keda",
        "repo": "https://github.com/kedacore/keda-docs.git",
        "branch": "main",
        "path": "content/docs/2.19",
    },
    {
        "name": "eks-best-practices",
        "repo": "https://github.com/aws/aws-eks-best-practices.git",
        "branch": "mainline",
        "path": "latest/bpg/autoscaling",
    },
]

BLOG_URLS = [
    # Karpenter
    "https://aws.amazon.com/blogs/containers/optimizing-your-kubernetes-compute-costs-with-karpenter-consolidation/",
    "https://aws.amazon.com/blogs/containers/announcing-karpenter-1-0/",
    "https://aws.amazon.com/blogs/containers/using-amazon-ec2-spot-instances-with-karpenter/",
    "https://aws.amazon.com/blogs/compute/applying-spot-to-spot-consolidation-best-practices-with-karpenter/",
    "https://aws.amazon.com/blogs/containers/how-grover-saves-costs-with-80-spot-in-production-using-karpenter-with-amazon-eks/",
    "https://aws.amazon.com/blogs/containers/migrating-from-x86-to-aws-graviton-on-amazon-eks-using-karpenter/",
    "https://aws.amazon.com/blogs/containers/how-to-upgrade-amazon-eks-worker-nodes-with-karpenter-drift/",
    "https://aws.amazon.com/blogs/containers/manage-scale-to-zero-scenarios-with-karpenter-and-serverless/",
    "https://aws.amazon.com/blogs/containers/harnessing-karpenter-transitioning-kafka-to-amazon-eks-with-aws-solutions/",
    "https://aws.amazon.com/blogs/containers/cordials-journey-implementing-bottlerocket-and-karpenter-in-amazon-eks/",
    # Spot
    "https://aws.amazon.com/blogs/compute/best-practices-for-handling-ec2-spot-instance-interruptions/",
    # Cost optimization
    "https://aws.amazon.com/blogs/containers/cost-optimization-for-kubernetes-on-aws/",
    # KEDA
    "https://aws.amazon.com/blogs/containers/using-keda-with-karpenter-to-scale-from-zero-on-amazon-eks/",
    "https://aws.amazon.com/blogs/containers/autoscaling-kubernetes-workloads-with-keda-using-amazon-managed-service-for-prometheus-metrics/",
    "https://aws.amazon.com/blogs/mt/autoscaling-kubernetes-workloads-with-keda-using-amazon-managed-service-for-prometheus-metrics/",
    "https://aws.amazon.com/blogs/mt/proactive-autoscaling-of-kubernetes-workloads-with-keda-using-metrics-ingested-into-amazon-cloudwatch/",
    "https://aws.amazon.com/blogs/containers/scalable-and-cost-effective-event-driven-workloads-with-keda-and-karpenter-on-amazon-eks/",
]

# --- Shared clients ---


def get_opensearch_client() -> OpenSearch:
    host = OPENSEARCH_URL.replace("http://", "").replace("https://", "")
    hostname, port = host.split(":") if ":" in host else (host, "9200")
    return OpenSearch(
        hosts=[{"host": hostname, "port": int(port)}],
        use_ssl=False,
        verify_certs=False,
    )


# --- Chunking ---

def chunk_text(text: str, source: str, section: str) -> list[dict]:
    """Split text into overlapping chunks by size, respecting markdown headers."""
    # Strip Hugo/Jekyll front matter
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            text = text[end + 3:].strip()

    chunks = []
    current_section = section
    current_text = ""

    # Match real markdown headings ("# " ... "###### "), not shebangs
    # (#!/bin/bash) or code comments (#foo), which would otherwise create tiny
    # junk chunks.
    heading_re = re.compile(r"^#{1,6}\s+\S")

    for line in text.split("\n"):
        if heading_re.match(line):
            if len(current_text.strip()) > 100:
                chunks.append({"text": current_text.strip(), "source": source, "section": current_section})
            current_section = line.lstrip("#").strip()
            current_text = line + "\n"
        else:
            current_text += line + "\n"
            if len(current_text) > CHUNK_SIZE:
                chunks.append({"text": current_text.strip(), "source": source, "section": current_section})
                current_text = current_text[-CHUNK_OVERLAP:]

    if len(current_text.strip()) > 50:
        chunks.append({"text": current_text.strip(), "source": source, "section": current_section})

    return chunks


# --- Data sources ---

def clone_and_chunk(source: dict, tmpdir: str) -> list[dict]:
    """Shallow clone a repo and chunk all markdown files."""
    dest = os.path.join(tmpdir, source["name"])
    print(f"  Cloning {source['repo']} ({source['branch']})...")
    Repo.clone_from(
        source["repo"], dest,
        branch=source["branch"],
        depth=1,
        single_branch=True,
    )
    docs_path = os.path.join(dest, source["path"])
    if not os.path.isdir(docs_path):
        print(f"    Warning: {source['path']} not found in repo")
        return []

    chunks = []
    for root, _, files in os.walk(docs_path):
        for f in files:
            if (f.endswith(".md") or f.endswith(".adoc")) and not f.startswith("_index"):
                filepath = os.path.join(root, f)
                with open(filepath) as fh:
                    text = fh.read()
                section = os.path.basename(filepath).replace(".md", "")
                chunks.extend(chunk_text(text, source["name"], section))

    return chunks


def fetch_blogs() -> list[dict]:
    """Fetch AWS blog posts and chunk them."""
    import requests
    from bs4 import BeautifulSoup

    chunks = []
    for url in BLOG_URLS:
        try:
            resp = requests.get(url, headers={"User-Agent": "slemify/1.0"}, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.find_all(["nav", "footer", "aside", "header"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.body
            if not main:
                continue
            text = main.get_text(separator="\n", strip=True)
            if len(text) < 200:
                continue
            slug = url.rstrip("/").split("/")[-1]
            blog_chunks = chunk_text(text, "aws-blog", slug)
            chunks.extend(blog_chunks)
            print(f"    {slug}: {len(blog_chunks)} chunks")
        except Exception as e:
            print(f"    Failed: {url} ({e})")
    return chunks


# --- Embedding and indexing ---

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via the in-cluster TEI pod (bge-base-en-v1.5)."""
    # TEI accepts a batch of inputs and returns one embedding per input.
    truncated = [t[:8000] for t in texts]
    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{EMBEDDING_URL}/embed", json={"inputs": truncated})
        resp.raise_for_status()
        return resp.json()


def index_chunks(client: OpenSearch, chunks: list[dict]) -> int:
    """Embed and index all chunks into OpenSearch in batches."""
    print(f"  Embedding and indexing {len(chunks)} chunks...")
    indexed = 0

    for i in range(0, len(chunks), 10):
        batch = chunks[i:i + 10]
        embeddings = embed_texts([c["text"] for c in batch])

        for chunk, embedding in zip(batch, embeddings):
            client.index(index=INDEX_NAME, body={**chunk, "embedding": embedding})
            indexed += 1

        if indexed % 50 == 0:
            print(f"    {indexed}/{len(chunks)} indexed")

    client.indices.refresh(index=INDEX_NAME)
    return indexed


def create_index(client: OpenSearch):
    """Create (or recreate) the k-NN index."""
    if client.indices.exists(index=INDEX_NAME):
        print(f"  Deleting existing index {INDEX_NAME}...")
        client.indices.delete(index=INDEX_NAME)

    client.indices.create(index=INDEX_NAME, body={
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "source": {"type": "keyword"},
                "section": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    # Embeddings are L2-normalized (unit vectors), so inner
                    # product ranks identically to cosine similarity. faiss
                    # supports innerproduct across OpenSearch versions, whereas
                    # cosinesimil + faiss is only valid from 2.19+.
                    "method": {"name": "hnsw", "space_type": "innerproduct", "engine": "faiss"},
                },
            }
        },
    })
    print(f"  Created index {INDEX_NAME}")


# --- Main ---

def main():
    print("=== Indexing Knowledge Base ===")

    global INDEX_NAME

    append = "--append" in sys.argv
    source_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--source="):
            source_filter = arg.split("=", 1)[1]
        elif arg.startswith("--index-name="):
            INDEX_NAME = arg.split("=", 1)[1]

    print(f"  Index: {INDEX_NAME}")

    client = get_opensearch_client()

    if not append:
        create_index(client)
    else:
        print("  Appending to existing index")

    # Determine which git sources to process
    sources = SOURCES
    if source_filter:
        if source_filter == "aws-blog":
            sources = []
        else:
            sources = [s for s in SOURCES if s["name"] == source_filter]
            if not sources:
                print(f"  Error: unknown source '{source_filter}'")
                print(f"  Available: {[s['name'] for s in SOURCES] + ['aws-blog']}")
                return

    all_chunks = []
    tmpdir = tempfile.mkdtemp()

    try:
        for source in sources:
            print(f"\n--- {source['name']} ---")
            chunks = clone_and_chunk(source, tmpdir)
            print(f"  {len(chunks)} chunks")
            all_chunks.extend(chunks)

        if not source_filter or source_filter == "aws-blog":
            print("\n--- AWS Blog Posts ---")
            blog_chunks = fetch_blogs()
            all_chunks.extend(blog_chunks)
            print(f"  {len(blog_chunks)} chunks from blogs")

        print(f"\n--- Indexing ---")
        print(f"  Total chunks: {len(all_chunks)}")
        n = index_chunks(client, all_chunks)
        print(f"\n=== Done: {n} chunks indexed ===")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
