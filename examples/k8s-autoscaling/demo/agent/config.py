"""Configuration and shared clients for the orchestrator.

All environment-driven settings and the long-lived clients (Bedrock, OpenSearch)
live here so the rest of the package imports them from one place.
"""
import os

import boto3
from opensearchpy import OpenSearch

# --- Seats: which model fills each role ---
# Every model-backed step in the agent is a "seat". Each seat can be filled by
# the frontier LLM on Bedrock or by a small model served on CPU in-cluster. The
# defaults are the CPU-first system. Setting every seat to its LLM value gives
# the monolith: one frontier model doing every step. Moving seats one at a time
# from LLM to CPU, and measuring after each move, is the point of the workshop
# and the migration path for a real deployment.
#
#   TRIAGE  = llm | classifier   who classifies intent (Bedrock, or the Slemify
#                                ONNX classifier; the intent heuristic follows)
#   EMBED   = bedrock | slemify  who embeds queries (Titan, or the tuned encoder;
#                                the OpenSearch index follows, see below)
#   RERANK  = off | on           whether the cross-encoder re-ranks candidates
#   ANALYST = llm | slm          who drafts the answer (Bedrock, or the CPU SLM)
#
# The gate is always the LLM: judging a draft needs a capable model, and that is
# true in every configuration. Invalid values fail fast at import.
def _seat(name: str, default: str, allowed: tuple[str, ...]) -> str:
    val = os.environ.get(name, default).strip().lower()
    if val not in allowed:
        raise ValueError(f"{name}={val!r} is not one of {allowed}")
    return val


TRIAGE = _seat("TRIAGE", "classifier", ("llm", "classifier"))
EMBED = _seat("EMBED", "slemify", ("bedrock", "slemify"))
RERANK = _seat("RERANK", "on", ("off", "on"))
ANALYST = _seat("ANALYST", "slm", ("llm", "slm"))


def seats() -> dict:
    """The current seat assignment, for /stats and the eval scorecard."""
    return {"triage": TRIAGE, "embed": EMBED, "rerank": RERANK, "analyst": ANALYST}


# --- Service endpoints ---
TRIAGE_URL = os.environ.get("TRIAGE_URL", "http://localhost:8081")
ANALYST_URL = os.environ.get("ANALYST_URL", "http://localhost:8082")
OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
# In-cluster embedding served by the Slemify-trained retriever (TEI /embed, 768d);
# the dimension must match index-knowledge.py's index mapping.
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:8083")
# In-cluster cross-encoder re-ranker.
RERANKER_URL = os.environ.get("RERANKER_URL", "http://localhost:8084")
# Sandbox that executes the cluster-touching tools in a separate pod. When set,
# the orchestrator holds NO cluster RBAC and calls this service over HTTP; the
# tools pod is the only workload with K8s credentials. Empty = run tools in-process
# (single-pod dev mode).
TOOLSVC_URL = os.environ.get("TOOLSVC_URL", "")
# Two indexes over the same corpus, one per embedding model, because the vector
# dimension is fixed at index time (768 for the tuned encoder, 1024 for Titan).
# The EMBED seat picks which one queries hit; index-knowledge.py builds either.
INDEX_NAME = os.environ.get("INDEX_NAME", "k8s-autoscaling-knowledge")
BEDROCK_INDEX_NAME = os.environ.get("BEDROCK_INDEX_NAME", "k8s-autoscaling-knowledge-bedrock")
ACTIVE_INDEX = BEDROCK_INDEX_NAME if EMBED == "bedrock" else INDEX_NAME

# --- Models ---
LLM_MODEL = os.environ.get("LLM_MODEL", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
# The faithfulness gate defaults to the same capable model as escalation: a small
# model proved too lenient at catching domain-specific wrong answers. Override
# GATE_MODEL to trade accuracy for cost.
GATE_MODEL = os.environ.get("GATE_MODEL", LLM_MODEL)
# Bedrock embedding model for EMBED=bedrock. Titan Text Embeddings v2 at 1024d;
# BEDROCK_EMBED_DIM must match the mapping index-knowledge.py wrote.
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
BEDROCK_EMBED_DIM = int(os.environ.get("BEDROCK_EMBED_DIM", "1024"))

# --- Cost accounting (the scoreboard) ---
# USD per 1M tokens, (input, output), matched by substring of the model id.
# List prices at time of writing; override with BEDROCK_PRICE_<NAME>_IN/OUT or
# add rows for other models. Only Bedrock is metered per query. CPU pods are
# capacity you pay for by the hour whether or not a query arrives, so their
# cost is reported separately (CPU_POOL_USD_PER_HOUR, informational) and never
# folded into the per-query number. Mixing the two would hide the difference
# between a variable cost and a fixed one, which is the actual trade-off.
_PRICE_TABLE = {
    "sonnet": (3.00, 15.00),
    "haiku": (0.80, 4.00),
    "titan-embed": (0.02, 0.0),
}
for _name in list(_PRICE_TABLE):
    _in = os.environ.get(f"BEDROCK_PRICE_{_name.upper().replace('-', '_')}_IN")
    _out = os.environ.get(f"BEDROCK_PRICE_{_name.upper().replace('-', '_')}_OUT")
    if _in or _out:
        _PRICE_TABLE[_name] = (float(_in or _PRICE_TABLE[_name][0]),
                               float(_out or _PRICE_TABLE[_name][1]))


def price_per_million(model_id: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens for a Bedrock model id; (0, 0) if unknown."""
    low = model_id.lower()
    for key, prices in _PRICE_TABLE.items():
        if key in low:
            return prices
    return (0.0, 0.0)


# Informational: what the CPU node pool costs per hour, for the worksheet.
CPU_POOL_USD_PER_HOUR = float(os.environ.get("CPU_POOL_USD_PER_HOUR", "0") or 0)

# --- Retrieval ---
# Wide candidate pools give the reranker a real set to choose from; dense (kNN)
# captures semantics, lexical (BM25) catches exact identifiers (API versions,
# field names). The reranker cuts the union down to the handful the analyst sees.
RETRIEVE_CANDIDATES = int(os.environ.get("RETRIEVE_CANDIDATES", "30"))
LEXICAL_CANDIDATES = int(os.environ.get("LEXICAL_CANDIDATES", "15"))
# The reranker only needs the question's intent; long config pastes blow up its
# latency, so cap the query it scores against (the full text still goes to the SLM).
RERANK_QUERY_CHARS = 512
# Docs kept after reranking (or after truncating vector order when RERANK=off),
# and extra pulled when a retry broadens the search.
KEEP_DOCS = int(os.environ.get("KEEP_DOCS", "5"))
BROADEN_EXTRA = 4

# --- Control flow ---
# Hard cap on tool calls per query so the plan -> tool loop can never run away.
MAX_TOOL_CALLS = 3
# Retries the critic may request before escalating.
MAX_CRITIC_RETRIES = 1

# --- Tools ---
# Read-only cluster tools are enabled by default; set TOOLS_ENABLED=false (or run
# without cluster credentials) to answer from documentation only.
TOOLS_ENABLED = os.environ.get("TOOLS_ENABLED", "true").lower() not in ("0", "false", "no")

# Write actions (remediation) are OFF unless ALLOW_APPLY is set AND the
# orchestrator has write RBAC. Even then, only a whitelist of deterministic,
# single-target patches is allowed (never free-form model YAML).
ALLOW_APPLY = os.environ.get("ALLOW_APPLY", "false").lower() in ("1", "true", "yes")

# Surface exactly what grounding the analyst receives (tools fired + assembled
# context) on the SSE stream, to tell a retrieval/routing gap from a generation gap.
DEBUG_CONTEXT = bool(os.environ.get("DEBUG_CONTEXT"))

# Prompt text lives in prompts.py (all instructions and templates in one place).

# --- Shared clients (initialized once) ---
bedrock = boto3.client("bedrock-runtime")


def _make_opensearch() -> OpenSearch:
    host = OPENSEARCH_URL.replace("http://", "").replace("https://", "")
    hostname, port = host.split(":") if ":" in host else (host, "9200")
    return OpenSearch(
        hosts=[{"host": hostname, "port": int(port)}],
        use_ssl=False,
        verify_certs=False,
    )


opensearch = _make_opensearch()
