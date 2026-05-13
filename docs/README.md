# Slemify Deep Dives

Technical deep dives into each stage of the Slemify pipeline. Written for Platform Engineers who want to understand not just *what* Slemify does, but *why* it makes the choices it does.

Each doc covers the design decisions, best practices, and research behind a pipeline stage. They're meant to be read by humans and consumed by agents.

## Getting Started

New to Slemify? Start here:

- [Getting Started: Build a Multi-Agent K8s Expert](getting-started.md). End-to-end tutorial that walks you through training two SLMs and wiring them into a multi-agent demo.

## Stages

| Stage | Doc | What it covers |
|-------|-----|----------------|
| Data | [data.md](deep-dive/data.md) | Raw data quality, synthetic generation, label taxonomy, class balance, independent evaluation |
| Training | [training.md](deep-dive/training.md) | QLoRA fine-tuning, model sizing, Spot GPU, checkpointing, loss curves |
| Serving | [serving.md](deep-dive/serving.md) | Reference deployment, CPU inference, latency optimization, autoscaling guidance |
| Report | [report.md](deep-dive/report.md) | Accuracy measurement, SLM vs LLM comparison, cost projections |

## Further Reading

- [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). why memory bandwidth (not FLOPs) determines inference speed
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (NVIDIA, 2025). the research case for SLMs in production
- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314). the fine-tuning technique Slemify uses
