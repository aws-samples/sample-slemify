"""Agentic k8s-autoscaling orchestrator, organized by responsibility.

Modules:
  config       env, constants, shared clients
  retrieval    embed -> hybrid search -> rerank
  generation   stream from the auditor SLM or the Bedrock LLM
  gate         faithfulness gate + deprecated-config lint
  tools        read-only Kubernetes evidence tools
  remediation  gated, bounded write actions (human-in-the-loop)
  classify     triage category + question intent
  extract      heuristic resource/argument extraction
  graph        intent-routed LangGraph orchestration
"""
