# K8s Autoscaling Example

A complete example of training and deploying domain-specific SLMs for Kubernetes autoscaling support using Slemify.

## Structure

```
k8s-autoscaling/
├── auditor/          # Auditor model config (8B generation, config analysis)
├── triage/           # Triage model config (encoder classifier, intent routing)
├── risk-scorer/      # Risk scorer config (encoder regression, 0.0-1.0 risk score)
├── retriever/        # Retriever config (embedding, domain-tuned RAG vectors)
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
| Triage | encoder (768d) | `classification` — intent routing + confidence | ~25ms |
| Risk Scorer | encoder (768d) | `scoring` — operational risk 0.0-1.0 | ~25ms |
| Retriever | encoder (768d) | `embedding` — domain-tuned RAG vectors | ~25ms |
| Auditor | 8B (q4_k_m) | `generation` — structured config analysis | ~14s streaming |

All run on Graviton CPUs with no GPU required for serving. The auditor is fine-tuned on GPU (QLoRA); the encoder-family models train on CPU — triage and risk scorer fit a head in seconds, the retriever contrastively fine-tunes in a few minutes. (The demo also runs a stock cross-encoder *reranker* on CPU — that's a serving pattern, not a Slemify-trained model; see the repo FAQ on why reranking isn't a task.)

## Quick Start

```bash
# Train and deploy the models you need
slemify train --config auditor/expert.yaml
slemify train --config triage/expert.yaml
slemify train --config risk-scorer/expert.yaml
slemify train --config retriever/expert.yaml
slemify deploy --config auditor/expert.yaml
slemify deploy --config triage/expert.yaml
slemify deploy --config risk-scorer/expert.yaml
slemify deploy --config retriever/expert.yaml

# Run the demo
cd demo && ./scripts/deploy.sh
```

See [demo/README.md](demo/README.md) for the full multi-agent demo with RAG and LLM fallback.

## Routing on a slice, not the whole input

The triage classifier has a small context window (the encoder caps at ~512
tokens, ~350-400 words). That is rarely a problem here, and often an advantage:
you scale a router by sending it **less, more often** — not more. A router only
needs the *decision-relevant* signal, not the entire input.

This demo uses small models exactly that way — each makes a cheap decision on a
small slice, and only the expensive model does the heavy lifting:
- **Route the query** — triage classifies the user's question/intent. A question is short by nature.
- **Guard the input** — a safety / PII / out-of-scope check before any LLM runs.
- **Filter retrieved chunks** — the reranker scores each retrieved chunk *one at a time*; each chunk is small, so you never need a big window — you scale by making many small calls, not one giant one.

If the routing signal is buried in a long document, don't grow the window —
shrink the input first: route on the latest message, the subject line, the first
paragraph, or a one-line summary.

Honest boundary: if a decision truly needs to read a whole long document at once
(say, judging the overall risk of a 30-page contract), a small classifier on the
raw text won't do it — summarize or extract first, chunk-and-aggregate, or point
`model.base` at a longer-context encoder. The small model is a scalpel, not a
bucket. See [serving.md](../../docs/deep-dive/serving.md) for the context-window details.
