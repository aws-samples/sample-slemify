# Slemify

Generate, train, and validate small specialist models. One YAML, one command.

Small specialist models handle the high-volume, repetitive tasks in your AI workflows (classification, routing, extraction) so your LLMs can focus on what they're best at. Slemify automates the path from raw data to a validated, production-ready model — a fine-tuned generative SLM (GGUF) for free-form tasks, or a CPU-trained encoder classifier for routing and labeling. How you deploy that model is up to you.

```bash
slemify deploy --config expert.yaml
```

Slemify picks the right model family from `project.task`:

- `task: generation` — a causal LM fine-tuned with QLoRA on GPU, served on CPU (GGUF/llama.cpp). For reasoning and free-form output.
- `task: classification` — a frozen encoder + a lightweight head, trained and served entirely on CPU in seconds. For routing, intent, and labeling.
- `task: scoring` — the same encoder-head family with a regression head, trained and served on CPU. Returns a single number in [0,1]. For risk/quality/confidence guardrails. (`extraction` and `reranking` also extend this family.)

## When to Use an SLM

Not every task needs a frontier model. Most agentic AI systems have "hot spots". repetitive sub-tasks that run thousands of times a day with the same pattern. These are ideal for a specialized SLM:

| Task Type | Example | Why SLM |
|-----------|---------|---------|
| Classification | Alert triage, intent routing, document categorization | Same pattern, different inputs. Fast, predictable output. |
| Scoring | Risk/quality/confidence guardrails on a config, answer, or request | One number in [0,1] decides auto-approve vs escalate. Cheap on every request. |
| Extraction | Pull structured fields from logs, invoices, clinical notes | Rigid output schema. Doesn't need world knowledge. |
| Routing | Pick which tool/API/agent handles a request | Binary or multi-class decision. Sub-100ms matters. |
| Validation | Safety checks, compliance gates, format verification | Rule-based logic baked into weights. Runs on every request. |

**The criteria:** high repetition, low semantic variation, structured output. If the task looks the same every time with different inputs, an SLM can do it faster and cheaper than a general-purpose LLM. often with higher accuracy for that specific task.

## SLMs + LLMs Together

Slemify doesn't replace LLMs. It adds a fast, cheap layer alongside them.

```
[Request] → [SLM Router] → high confidence → [SLM Result] → done (50ms, $0)
                          → low confidence  → [LLM Fallback] → done (3s, $0.01)
```

The inference endpoint exposes an OpenAI-compatible API (`/v1/chat/completions`). Any agent, orchestrator, or application can call it directly via HTTP. Set `llm_endpoint` in your config to any OpenAI-compatible API (vLLM, llama.cpp, Bedrock proxy) for LLM fallback. The SLM handles 70-90% of requests at fixed cost. The LLM handles the rest.

| Architecture | Cost at 10K requests/day | Avg Latency |
|-------------|------------------------|-------------|
| 100% LLM API | ~$3,000/mo | 1-3s |
| SLM + LLM (90/10) | ~$500/mo | 200ms avg |

## How It Works

```
expert.yaml → [DATA] → [TRAINING] → [SERVING + VALIDATION]
                 │          │                   │
            Ingest +    QLoRA on          Deploy model,
            Synthetic   Spot GPU          run eval report,
            via Bedrock via Unsloth       generate HTML
```

1. **Data**. Ingests your raw data from S3. Bedrock generates synthetic training pairs from your source content. You verify the output before training.
2. **Training**. QLoRA fine-tuning on Spot GPU via Unsloth. Exports a quantized GGUF model to S3.
3. **Serving + Validation**. Deploys the model on a live endpoint, runs the evaluation dataset through it, and generates an HTML report with accuracy, latency, and cost projections.

The output is a GGUF model file in S3 and a production readiness report. The serving deployment that Slemify creates is production-quality and serves as a reference for your own infrastructure. You can use it as-is, adapt it, or serve the GGUF with any compatible runtime (llama.cpp, vLLM, Ollama). See the [Serving deep dive](docs/deep-dive/serving.md) for deployment guidance and best practices.

## Quick Start

### Prerequisites

- EKS cluster with [Karpenter](https://karpenter.sh)
- S3 bucket for data and artifacts
- AWS credentials with Bedrock access
- `kubectl` configured for your cluster

### 1. Define your task

```yaml
apiVersion: slemify/v1

project:
  name: k8s-autoscaling-triage
  task: classification
  domain: >
    Classify Kubernetes autoscaling support queries into a routing
    category. Each message is classified into exactly one category:
    karpenter_config, keda_config, hpa_config, pdb_disruption,
    spot_interruption, multi_resource, or noise for off-topic messages.
  labels:
    routing:
      - karpenter_config
      - keda_config
      - hpa_config
      - pdb_disruption
      - spot_interruption
      - multi_resource
      - noise

model:
  base: ""       # encoder model ID (e.g. BAAI/bge-base-en-v1.5)
  head: logistic # classifier head: logistic | linear | mlp

data:
  bucket: slemify-data
  path: k8s-autoscaling/data/
  sources:
    - path: queries/
      type: raw
  synthetic:
    model: eu.anthropic.claude-sonnet-4-6
    pairs: 1200
  evaluation:
    model: eu.anthropic.claude-sonnet-4-6
    pairs: 150
    sources:
      - path: eval-queries/
        type: raw

training:
  spot: true
```

### 2. Upload your training data

```bash
aws s3 sync ./data/queries s3://slemify-data/k8s-autoscaling/data/queries/
aws s3 sync ./data/eval-queries s3://slemify-data/k8s-autoscaling/data/eval-queries/
```

### 3. Deploy

```bash
slemify deploy --config expert.yaml
```

Slemify handles data processing, synthetic pair generation, training, quantization, and validation. The resulting GGUF model is uploaded to S3. You then deploy it in your own infrastructure using the reference deployment as a starting point.

### 4. View the report

```bash
slemify report --config expert.yaml
```

Downloads the HTML report from S3 and opens it in your browser. The report includes accuracy metrics, latency benchmarks, SLM vs LLM comparison, and cost projections.

## How Much Data Do I Need?

| Task Type | Training Examples | Notes |
|-----------|------------------|-------|
| Classification (routing, triage) | 200-500 | Binary or multi-class. Clear categories. |
| Scoring (risk, quality, confidence) | 500-1,200 | Regression target in [0,1]. Spread examples across the full range. |
| Extraction (fields from text) | 500-1,000 | More examples = better edge case coverage. |
| Structured generation (commands, configs) | 500-1,000 | Model needs to learn output format precisely. |

Quality matters more than quantity. 500 well-curated instruction-response pairs beat 10,000 noisy ones. Bedrock generates synthetic examples from your source data, so you don't need to write them all by hand.

## Cost

| Item | Cost |
|------|------|
| Training (Spot GPU, one-time) | ~$0.15 |
| Synthetic data (Bedrock) | ~$10-50 |
| **Total to generate a model** | **~$15-55** |

Inference cost depends on how you deploy. The reference deployment (llama.cpp on CPU Spot) runs at ~$117/mo per replica. Throughput scales linearly: 3 replicas = 3x throughput at 3x cost. No rate limits, no per-token charges. See the [Serving deep dive](docs/deep-dive/serving.md) for cost comparisons across CPU, GPU, and LLM API options.

## Examples

- [K8s Autoscaling Auditor](examples/k8s-autoscaling/). Tiered SLM system: a triage classifier routes queries, an 8B auditor produces structured reasoning about Karpenter/KEDA/HPA misconfigurations
- [K8s Autoscaling Risk Scorer](examples/k8s-autoscaling/risk-scorer/). A `task: scoring` encoder-head model that rates a config change's operational risk 0.0–1.0 on CPU — a cheap guardrail that auto-approves low-risk changes and escalates high-risk ones to the auditor

## Deep Dives

Technical docs covering the design decisions, best practices, and research behind each pipeline stage. Written for Platform Engineers.

- [Getting Started](docs/getting-started.md). End-to-end tutorial: build a multi-agent K8s expert from scratch
- [Data Stage](docs/deep-dive/data.md). Raw data quality, synthetic generation, label taxonomy, verification
- [Training Stage](docs/deep-dive/training.md). QLoRA, model sizing, Spot GPU, checkpointing, quantization
- [Serving Stage](docs/deep-dive/serving.md). Reference deployment, CPU inference, autoscaling guidance
- [Report Stage](docs/deep-dive/report.md). Accuracy measurement, SLM vs LLM comparison, cost projections

## Architecture

The pipeline runs on Kubernetes (EKS). The output is a GGUF model in S3.

- **Karpenter**. GPU nodes for training (Spot), CPU nodes for the reference deployment
- **Unsloth**. QLoRA fine-tuning, 2-5x faster than standard training
- **llama.cpp**. GGUF inference on CPU (used in the reference deployment and validation report)
- **Pod Identity**. IAM access to S3 and Bedrock, no static credentials
- **Systems Manager**. Remote container builds via SSM, no SSH keys or open ports required

The reference serving deployment (llama.cpp on CPU) is included for validation and as a starting point. You can serve the GGUF model with any compatible runtime: llama.cpp, vLLM, Ollama, or any tool that reads GGUF files.

## Commands

| Command | Description |
|---------|-------------|
| `slemify deploy` | Run the full pipeline |
| `slemify deploy --stage training --no-wait` | Submit a stage and exit |
| `slemify status my-project` | Show pipeline progress |
| `slemify status my-project -o json` | Machine-readable status for agents |
| `slemify validate` | Validate config without deploying |
| `slemify report` | Download and open the accuracy report in the browser |
| `slemify report --output my-report.html` | Save report to a custom path |
| `slemify report --no-open` | Download without opening the browser |
| `slemify build` | Build container images to ECR |

## FAQ

**Q: When should I use an SLM vs just calling an LLM API?**
A: If the task is repetitive, structured, and runs more than ~1,000 times/day. or if data can't leave your VPC. Below that volume, an LLM API is simpler and fine.

**Q: Can a 3B model really match a frontier LLM?**
A: For general tasks, no. For YOUR specific structured task with YOUR categories, a fine-tuned 3B model matches or beats general-purpose LLMs. [Salesforce's xLAM-2-8B](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r) beat GPT-4o and Claude 3.5 at tool calling on the [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html). Specialization beats size.

**Q: What about RAG?**
A: SLMs and RAG solve different problems. RAG retrieves relevant context for knowledge questions. SLMs handle classification, routing, and extraction where you don't need retrieval. you need a fast decision. They work well together: SLM routes the query, RAG handles the knowledge lookup.

**Q: Can I use a different base model?**
A: Yes. For `task: generation`, any HuggingFace causal LM supported by Unsloth. For encoder-head tasks (`task: classification`, `task: scoring`), any sentence-transformers encoder (e.g. `BAAI/bge-base-en-v1.5`). The auto-sizer adjusts infrastructure based on the task and model size.

**Q: What happens during a Spot interruption?**
A: Training checkpoints sync to S3 every 500 steps. The next pod resumes from the last checkpoint automatically. Max work lost: ~20 minutes.

## Agent Skill

Slemify includes an [agent skill](skills/slemify/SKILL.md) compatible with Claude Code, OpenAI Codex, Gemini CLI, and Cursor. The skill teaches AI coding agents how to identify SLM opportunities in your system, design the agent's role, write the expert.yaml config, run the pipeline, and interpret results.

Install in Claude Code:
```bash
/plugin install slemify@<your-repo>
```

Or reference the skill directly:
```
"Use the Slemify skill to identify which of my LLM calls could be replaced with a specialized SLM."
```

The skill includes templates for two patterns:
- **Router Agent** (`task: classification`): a CPU encoder classifier for fast routing and intent decisions
- **Analyst Agent** (`task: generation`, 7-8B): structured reasoning grounded by RAG

## References

- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (NVIDIA, 2025). Position paper arguing SLMs under 10B parameters can handle 60-80% of agentic AI tasks
- [xLAM: Large Action Models](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r) (Salesforce). 8B model that beat GPT-4o at tool calling, proving specialization beats size
- [Forbes: Don't Default to the Biggest AI Model](https://www.forbes.com/councils/forbestechcouncil/2026/04/22/dont-default-to-the-biggest-ai-model-agentic-systems-deserve-better/). 40-70% of agentic AI invocations can use SLMs
- [Hallucination Propensity in Small Models](https://arxiv.org/abs/2411.00878). Research on knowledge mismatch between fine-tuning data and base model knowledge
- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314). The fine-tuning technique Slemify uses
- [Unsloth](https://github.com/unslothai/unsloth). 2-5x faster QLoRA training with custom Triton kernels
- [llama.cpp](https://github.com/ggerganov/llama.cpp). GGUF inference engine for CPU deployment
- [Model Context Protocol](https://modelcontextprotocol.io). How SLMs expose tools to AI assistants
