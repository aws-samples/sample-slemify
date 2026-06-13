# Slemify Deep Dives

Technical deep dives into each stage of the Slemify pipeline. Written for Platform Engineers who want to understand not just *what* Slemify does, but *why* it makes the choices it does.

Each doc covers the design decisions, best practices, and research behind a pipeline stage. They're meant to be read by humans and consumed by agents.

## Getting Started

New to Slemify? Start here:

- [Getting Started: Build a Multi-Agent K8s Expert](getting-started.md). End-to-end tutorial that walks you through training two SLMs and wiring them into a multi-agent demo.

## Task families

Slemify produces two kinds of specialist model, selected by `project.task`:

- **Generation** (`task: generation`) — a causal LM fine-tuned with QLoRA on
  GPU, exported to GGUF, served on CPU via llama.cpp. For free-form output:
  reasoning, audit reports, structured generation.
- **Encoder-head** (`task: classification` and `task: scoring`, with `extraction`
  and `reranking` to follow) — a frozen encoder plus a lightweight head, trained
  and served entirely on CPU. Classification emits a label + confidence; scoring
  emits a single number in [0,1]. For routing, intent, guardrails, and risk/quality
  scores.

The deep dives below note where the two paths differ.

## Stages

| Stage | Doc | What it covers |
|-------|-----|----------------|
| Data | [data.md](deep-dive/data.md) | Raw data quality, synthetic generation, label taxonomy, class balance, independent evaluation |
| Training | [training.md](deep-dive/training.md) | Generation: QLoRA fine-tuning, sizing, Spot GPU, GGUF. Encoder-head: frozen-encoder + head fit on CPU. Embedding: contrastive fine-tune on CPU |
| Serving | [serving.md](deep-dive/serving.md) | Generation: llama.cpp + GGUF. Encoder family: encoder + head/vector via ONNX. CPU inference, latency, autoscaling |
| Report | [report.md](deep-dive/report.md) | Generation: LLM-as-judge. Classification: accuracy + per-class P/R/F1. Scoring: MAE/R². Embedding: recall@k/MRR. Cost projections |

## Further Reading

- [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). why memory bandwidth (not FLOPs) determines inference speed
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (NVIDIA, 2025). the research case for SLMs in production
- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314). the fine-tuning technique Slemify uses
