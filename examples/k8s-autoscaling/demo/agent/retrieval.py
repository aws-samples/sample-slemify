"""RAG retrieval: embed the query, search the knowledge base (hybrid dense +
lexical), and re-rank the candidates down to the few docs the model sees.

Two seats live here (see config.py): EMBED decides who turns the query into a
vector (Bedrock Titan or the Slemify-tuned encoder on CPU) and therefore which
index is searched; RERANK decides whether the cross-encoder re-orders the
candidate pool or vector order is kept as-is.

Pure functions over the shared OpenSearch client; the graph composes them and
emits step events between stages.
"""
import json

import httpx

from . import config


def _embed_slemify(text: str) -> list[float]:
    """The Slemify-trained retriever (TEI /embed, 768d)."""
    with httpx.Client(timeout=10) as client:
        resp = client.post(f"{config.EMBEDDING_URL}/embed", json={"inputs": text[:8000]})
        resp.raise_for_status()
        return resp.json()[0]  # TEI returns one embedding per input


def _embed_bedrock(text: str) -> list[float]:
    """Titan Text Embeddings v2 via Bedrock (normalized, 1024d by default)."""
    resp = config.bedrock.invoke_model(
        modelId=config.BEDROCK_EMBED_MODEL,
        body=json.dumps({"inputText": text[:8000], "dimensions": config.BEDROCK_EMBED_DIM,
                         "normalize": True}),
        contentType="application/json", accept="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]


def embed_query(text: str) -> list[float]:
    """Embed the query with whoever holds the EMBED seat."""
    return _embed_bedrock(text) if config.EMBED == "bedrock" else _embed_slemify(text)


def _fmt_hit(hit: dict) -> str:
    """Format an OpenSearch hit as a labeled chunk. The full chunk text is kept
    (chunks are bounded at index time); an earlier 500-char cap silently
    decapitated chunks so facts past char 500 never reached the model."""
    s = hit["_source"]
    return f"[{s.get('source', '')} / {s.get('section', '')}]\n{s['text']}"


def vector_search(embedding: list[float], k: int) -> list[str]:
    """k-NN (dense / semantic) search over the indexed corpus."""
    res = config.opensearch.search(
        index=config.ACTIVE_INDEX,
        body={
            "size": k,
            "query": {"knn": {"embedding": {"vector": embedding, "k": k}}},
            "_source": ["text", "source", "section"],
        },
    )
    return [_fmt_hit(h) for h in res["hits"]["hits"]]


def keyword_search(query: str, k: int) -> list[str]:
    """BM25 (lexical) search — matches exact terms/identifiers (API versions,
    field names) that dense vectors can blur. Unioned with the vector pool."""
    if not query.strip():
        return []
    try:
        res = config.opensearch.search(
            index=config.ACTIVE_INDEX,
            body={
                "size": k,
                "query": {"match": {"text": query}},
                "_source": ["text", "source", "section"],
            },
        )
        return [_fmt_hit(h) for h in res["hits"]["hits"]]
    except Exception as e:
        print(f"  Lexical search failed, vector-only: {e}")
        return []


def hybrid_candidates(embedding: list[float], query: str, broaden: bool = False) -> list[str]:
    """Union dense + lexical candidates, de-duplicated, dense-first. The reranker
    reorders by relevance so merge order only affects ties."""
    extra = config.BROADEN_EXTRA if broaden else 0
    vec = vector_search(embedding, config.RETRIEVE_CANDIDATES + extra)
    lex = keyword_search(query, config.LEXICAL_CANDIDATES + extra)
    seen, out = set(), []
    for c in vec + lex:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# Field-aware retrieval: official-docs sources whose chunks define API fields
# (as opposed to blogs/examples that merely use them).
_AUTHORITATIVE_SOURCES = ["karpenter", "keda"]


def field_definition_docs(fields: list[str], exclude: list[str],
                          max_fields: int = 6, per_field: int = 1) -> list[str]:
    """Fetch the authoritative definition chunk for each named field.

    Complements (does not replace) the reranked global top-k: for a pasted
    multi-field config, similarity ranking favors chunks that look like the
    query — other example YAML, blog posts — and can drop the terse reference
    chunk that actually defines one of the fields. The model then fabricates
    about exactly that field (proven: consolidateAfter, lessons-learned
    section 20 — the defining chunk existed in the corpus, scored 2.9, and
    never reached the model). A targeted per-field lexical search filtered to
    the official docs reliably ranks the definition first (validated against
    the live corpus before this was written).

    Best-effort: any search failure returns what was found so far — the
    reranked docs still ground the answer, just without the guarantee."""
    out = []
    seen = set(exclude)
    for field in fields[:max_fields]:
        try:
            res = config.opensearch.search(
                index=config.ACTIVE_INDEX,
                body={
                    "size": per_field,
                    "query": {"bool": {
                        "must": {"match": {"text": field}},
                        "filter": {"terms": {"source": _AUTHORITATIVE_SOURCES}},
                    }},
                    "_source": ["text", "source", "section"],
                },
            )
            for h in res["hits"]["hits"]:
                doc = _fmt_hit(h)
                if doc not in seen:
                    seen.add(doc)
                    out.append(doc)
        except Exception as e:
            print(f"  Field-definition lookup failed for '{field}': {e}")
            break
    return out


def rerank_docs(query: str, docs: list[str], top_k: int) -> list[str]:
    """Re-rank candidates with the cross-encoder, keeping the best top_k. With
    RERANK=off, keep vector order truncated to top_k (the monolith's retrieval).
    Also falls back to vector order if the reranker is unavailable."""
    if not docs:
        return []
    if config.RERANK == "off":
        return docs[:top_k]
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{config.RERANKER_URL}/rerank",
                json={"query": query[:config.RERANK_QUERY_CHARS], "documents": docs, "top_k": top_k},
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        return [docs[r["index"]] for r in results]
    except Exception as e:
        print(f"  Rerank failed, using vector order: {e}")
        return docs[:top_k]
