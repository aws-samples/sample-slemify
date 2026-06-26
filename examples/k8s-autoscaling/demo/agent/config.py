"""Configuration and shared clients for the orchestrator.

All environment-driven settings and the long-lived clients (Bedrock, OpenSearch)
live here so the rest of the package imports them from one place.
"""
import os

import boto3
from opensearchpy import OpenSearch

# --- Service endpoints ---
TRIAGE_URL = os.environ.get("TRIAGE_URL", "http://localhost:8081")
AUDITOR_URL = os.environ.get("AUDITOR_URL", "http://localhost:8082")
OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
# In-cluster embedding served by the Slemify-trained retriever (TEI /embed, 768d);
# the dimension must match index-knowledge.py's index mapping.
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:8083")
# In-cluster cross-encoder re-ranker.
RERANKER_URL = os.environ.get("RERANKER_URL", "http://localhost:8084")
INDEX_NAME = os.environ.get("INDEX_NAME", "k8s-autoscaling-knowledge")

# --- Models ---
LLM_MODEL = os.environ.get("LLM_MODEL", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
# The faithfulness gate defaults to the same capable model as escalation: a small
# model proved too lenient at catching domain-specific wrong answers. Override
# GATE_MODEL to trade accuracy for cost.
GATE_MODEL = os.environ.get("GATE_MODEL", LLM_MODEL)

# --- Retrieval ---
# Wide candidate pools give the reranker a real set to choose from; dense (kNN)
# captures semantics, lexical (BM25) catches exact identifiers (API versions,
# field names). The reranker cuts the union down to the handful the auditor sees.
RETRIEVE_CANDIDATES = int(os.environ.get("RETRIEVE_CANDIDATES", "30"))
LEXICAL_CANDIDATES = int(os.environ.get("LEXICAL_CANDIDATES", "15"))
# The reranker only needs the question's intent; long config pastes blow up its
# latency, so cap the query it scores against (the full text still goes to the SLM).
RERANK_QUERY_CHARS = 512
# Docs kept after reranking, and extra pulled when a retry broadens the search.
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

# Surface exactly what grounding the auditor receives (tools fired + assembled
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
