# K8s Autoscaling Example

A complete example of training and deploying domain-specific SLMs for Kubernetes autoscaling support using Slemify.

## Structure

```
k8s-autoscaling/
├── auditor/          # Auditor model config (8B, config analysis)
├── triage/           # Triage model config (4B, intent classification)
├── data/
│   ├── queries/      # 76 training queries (real-world K8s configs)
│   └── eval-queries/ # 14 held-out evaluation queries
├── demo/             # Multi-agent demo application
│   ├── server.py     # FastAPI orchestrator (triage + RAG + auditor + LLM fallback)
│   ├── scripts/      # Helper scripts (deploy, indexing, tmux dashboard)
│   ├── Dockerfile    # Orchestrator container image
│   └── k8s-manifest.yaml
└── upload-to-s3.sh   # Upload training data to S3
```

## Models

| Model | Base | Task | Latency |
|-------|------|------|---------|
| Triage | 4B (q4_k_m) | Intent classification + confidence | ~1.5s |
| Auditor | 8B (q4_k_m) | Structured config analysis | ~14s streaming |

Both run on Graviton4 CPUs (c8g.4xlarge) with no GPU required.

## Quick Start

```bash
# Train and deploy both models
slemify train --config auditor/expert.yaml
slemify train --config triage/expert.yaml
slemify deploy --config auditor/expert.yaml
slemify deploy --config triage/expert.yaml

# Run the demo
cd demo && ./scripts/deploy.sh
```

See [demo/README.md](demo/README.md) for the full multi-agent demo with RAG and LLM fallback.
