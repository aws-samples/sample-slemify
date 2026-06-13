# K8s Autoscaling Example

A complete example of training and deploying domain-specific SLMs for Kubernetes autoscaling support using Slemify.

## Structure

```
k8s-autoscaling/
├── auditor/          # Auditor model config (8B generation, config analysis)
├── triage/           # Triage model config (encoder classifier, intent routing)
├── risk-scorer/      # Risk scorer config (encoder regression, 0.0-1.0 risk score)
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
| Triage | encoder (bge-base, 768d) | `classification` — intent routing + confidence | ~25ms |
| Risk Scorer | encoder (bge-base, 768d) | `scoring` — operational risk 0.0-1.0 | ~25ms |
| Auditor | 8B (q4_k_m) | `generation` — structured config analysis | ~14s streaming |

All run on Graviton CPUs with no GPU required for serving. The auditor is fine-tuned on GPU (QLoRA); the encoder-head models (triage, risk scorer) train on CPU in seconds.

## Quick Start

```bash
# Train and deploy the models you need
slemify train --config auditor/expert.yaml
slemify train --config triage/expert.yaml
slemify train --config risk-scorer/expert.yaml
slemify deploy --config auditor/expert.yaml
slemify deploy --config triage/expert.yaml
slemify deploy --config risk-scorer/expert.yaml

# Run the demo
cd demo && ./scripts/deploy.sh
```

See [demo/README.md](demo/README.md) for the full multi-agent demo with RAG and LLM fallback.
