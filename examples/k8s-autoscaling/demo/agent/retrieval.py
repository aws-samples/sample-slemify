"""RAG retrieval: embed the query, search the knowledge base (hybrid dense +
lexical), and re-rank the candidates down to the few docs the model sees.

Pure functions over the shared OpenSearch client; the graph composes them and
emits step events between stages.
"""
import httpx

from . import config


def embed_query(text: str) -> list[float]:
    """Embed text via the Slemify-trained retriever (TEI /embed, 768d)."""
    with httpx.Client(timeout=10) as client:
        resp = client.post(f"{config.EMBEDDING_URL}/embed", json={"inputs": text[:8000]})
        resp.raise_for_status()
        return resp.json()[0]  # TEI returns one embedding per input


def _fmt_hit(hit: dict) -> str:
    """Format an OpenSearch hit as a labeled chunk. The full chunk text is kept
    (chunks are bounded at index time); an earlier 500-char cap silently
    decapitated chunks so facts past char 500 never reached the model."""
    s = hit["_source"]
    return f"[{s.get('source', '')} / {s.get('section', '')}]\n{s['text']}"


def vector_search(embedding: list[float], k: int) -> list[str]:
    """k-NN (dense / semantic) search over the indexed corpus."""
    res = config.opensearch.search(
        index=config.INDEX_NAME,
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
            index=config.INDEX_NAME,
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


def rerank_docs(query: str, docs: list[str], top_k: int) -> list[str]:
    """Re-rank candidates with the cross-encoder, keeping the best top_k. Falls
    back to vector order (truncated) if the reranker is unavailable."""
    if not docs:
        return []
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
