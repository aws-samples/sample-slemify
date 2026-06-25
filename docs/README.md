# Slemify Deep Dives

Technical deep dives into each stage of the Slemify pipeline. Written for Platform Engineers who want to understand not just *what* Slemify does, but *why* it makes the choices it does.

Each doc covers the design decisions, best practices, and research behind a pipeline stage. They're meant to be read by humans and consumed by agents.

## Getting Started

New to Slemify? Start here:

- [Getting Started: Build a Multi-Agent K8s Expert](getting-started.md). End-to-end tutorial that walks you through training two SLMs and wiring them into a multi-agent demo.

## Task families

Slemify produces a few families of specialist model, selected by `project.task`:

- **Generation** (`task: generation`) — a causal LM served stock: downloaded,
  converted to GGUF, and quantized on CPU, served via llama.cpp and grounded by
  RAG. No fine-tuning. For free-form output: reasoning, audit reports, structured
  generation.
- **Encoder-head** (`task: classification` and `task: scoring`, with `extraction`
  to follow) — a frozen encoder plus a lightweight head, trained
  and served entirely on CPU. Classification emits a label + confidence; scoring
  emits a single number in [0,1]. For routing, intent, guardrails, and risk/quality
  scores.
- **Embedding** (`task: embedding`) — a bi-encoder fine-tuned for search and
  served on CPU (ONNX). It emits a vector for first-stage retrieval (RAG), and is
  domain-tuned over your own corpus.

Reranking (a cross-encoder that scores query/document pairs jointly) is a
deliberate non-goal: a strong stock cross-encoder is already well-calibrated and
fine-tuning it on synthetic data degrades it, so Slemify doesn't build one.
Running a stock cross-encoder reranker on CPU is shown as a serving pattern in
the k8s-autoscaling demo instead.

The deep dives below note where the two paths differ.

## Stages

| Stage | Doc | What it covers |
|-------|-----|----------------|
| Data | [data.md](deep-dive/data.md) | Raw data quality, synthetic generation, label taxonomy, class balance, independent evaluation |
| Training | [training.md](deep-dive/training.md) | Generation: served stock (download → GGUF → quantize) on CPU, no fine-tuning. Encoder-head: frozen-encoder + head fit on CPU. Embedding: contrastive fine-tune on CPU |
| Serving | [serving.md](deep-dive/serving.md) | Generation: llama.cpp + GGUF. Encoder family: encoder + head/vector via ONNX. CPU inference, latency, autoscaling |
| Report | [report.md](deep-dive/report.md) | Generation: LLM-as-judge. Classification: accuracy + per-class P/R/F1. Scoring: MAE/R². Embedding: recall@k/MRR. Cost projections |

## Further Reading

- [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). why memory bandwidth (not FLOPs) determines inference speed
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (NVIDIA, 2025). the research case for SLMs in production
- [Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning](https://arxiv.org/abs/2505.11574). why low-bit quantization hurts reasoning more than perplexity
