# Getting Started: Build a Multi-Agent K8s Expert

This guide walks you through building a complete multi-agent system using Slemify. By the end, you'll have two specialist models running on CPUs that audit Kubernetes autoscaling configurations, backed by a RAG knowledge base and an LLM fallback for edge cases. One is a CPU-trained encoder classifier (triage), the other a stock generative SLM served on CPU and grounded by RAG (auditor).

Total time: ~1 hour (mostly waiting for the convert and indexing steps). Cost: ~$15-30 (mostly Bedrock synthetic data for the triage classifier; the auditor is convert-only on CPU).

## The problem

Teams paste Kubernetes configs into ChatGPT and get answers that look correct but contain subtle errors. An LLM confidently tells you that `minValues: 3` in a Karpenter NodePool "blocks scheduling until 3 different families are provisioned" (wrong). It suggests `m6i` with `instance-generation Gt 6` (contradictory). It recommends deprecated APIs for current cluster versions.

These hallucinations are hard to catch because the output is well-formatted and sounds authoritative. A small model grounded in the actual API docs via RAG doesn't have this problem. It answers from the retrieved source of truth instead of from memory, and a faithfulness gate rejects any draft the docs don't support.

## What we'll build

```
User question or YAML config
        |
        v
[Triage classifier - encoder + head, CPU, ~25ms]
  Routes to the right handler
        |
        |-- noise --> rejected
        |-- low confidence --> LLM API + RAG
        |-- high confidence --> Auditor SLM + RAG
                                    |
                                    v
                            Structured analysis
                            (error type, severity, fix)
```

Two specialists, one pipeline — and they use two different model families:
- **Triage** (`task: classification`): a frozen encoder + logistic head that
  classifies intent in ~25ms. CPU-trained in seconds, deterministic.
- **Auditor** (`task: generation`): an 8B causal LM served stock and grounded by
  RAG that produces structured config analysis, streamed.

Both run on Graviton4 CPUs at inference time, and no GPU is used anywhere in the
pipeline: the triage classifier trains on CPU and the auditor is downloaded,
converted to GGUF, and quantized on CPU. The LLM API is called only for the
~10-20% of queries where triage isn't confident.

## Prerequisites

- EKS cluster with [Karpenter](https://karpenter.sh) installed
- S3 bucket for data and artifacts
- AWS credentials with Bedrock, S3, and EKS access
- `kubectl` configured for your cluster
- Slemify CLI installed (`go install` or download from releases)

## Step 1: Prepare training data

Training data is the foundation. For this example, we use real-world Kubernetes autoscaling questions, the kind that show up in Slack channels and support tickets.

Create a directory with `.txt` files, each containing one query:

```bash
mkdir -p data/queries
```

A good training query includes context and a YAML config:

```
# data/queries/karpenter-limits-01.txt
pods are stuck in Pending but karpenter isn't launching new nodes.
kubectl get nodepool shows we're at 48 CPU used out of 50 limit.

apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: default
spec:
  limits:
    cpu: "50"
    memory: 200Gi

should we increase the limits? what's a safe value?
```

You need 50-100 queries covering the scenarios you want the model to handle. Include:
- Valid configs (the model needs to learn what "correct" looks like)
- Common mistakes (deprecated APIs, conflicting constraints, missing fields)
- Edge cases (ambiguous questions, forwarded messages, off-topic noise)

See `examples/k8s-autoscaling/data/queries/` for 76 real examples.

Upload to S3:

```bash
aws s3 sync ./data/queries s3://slemify-data/k8s-autoscaling/data/queries/
```

## Step 2: Define the triage model

The triage model is a fast classifier. It reads the query and routes it to the
right category. Because routing is a closed-set classification problem, it uses
Slemify's **classification** task (`task: classification`) — a frozen encoder
plus a lightweight trained head, not a generative model. This trains on CPU in
seconds and serves deterministically in ~25ms. Create `triage/expert.yaml`:

```yaml
apiVersion: slemify/v1

project:
  name: k8s-autoscaling-triage
  task: classification
  domain: >
    Classify Kubernetes autoscaling support queries into a routing
    category.
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
  base: ""        # encoder model ID (a text encoder for classification)
  head: logistic  # classifier head: logistic | linear | mlp

data:
  bucket: slemify-data
  path: k8s-autoscaling/data/
  sources:
    - path: queries/
      type: raw
  synthetic:
    model: eu.anthropic.claude-sonnet-4-6
    pairs: 1200

training:
  spot: true
```

Key choices:
- **`task: classification`**: routes through the encoder-head path. No GPU, no
  GGUF — the encoder is frozen and only a small classifier head is trained.
- **encoder base** (a general-purpose text encoder): turns each query into a
  768-dim vector. A single pretrained encoder serves any classification domain.
- **`head: logistic`**: a logistic-regression head over the embeddings. Fast to
  fit, deterministic, and emits a calibrated confidence (the softmax
  probability) used for routing to the LLM fallback.
- **no `confidence` label**: confidence is the model's probability, not a
  predicted label, so it isn't part of the taxonomy.
- **1200 synthetic pairs**: Bedrock generates `query → category` examples from
  your raw queries.

## Step 3: Define the auditor model

The auditor is the expert. It receives queries that passed triage and produces structured analysis. Create `auditor/expert.yaml`:

```yaml
apiVersion: slemify/v1

project:
  name: k8s-autoscaling-auditor
  task: generation
  output_format: free_form
  domain: >
    Expert auditor for Kubernetes autoscaling configurations on EKS.
    Produces structured reasoning about what is wrong, why it is
    dangerous, and how to fix it.

model:
  base: ""  # HuggingFace causal LM (8B recommended for structured reasoning)
  # q8_0 held accuracy on this reasoning task in our eval; smaller quants lost
  # calibration. Re-check your own scorecard before going lower.
  quantize: q8_0

data:
  # Generation is served stock and grounded by RAG, so there is no synthetic data
  # or label taxonomy here. bucket/path are where the converted GGUF is uploaded.
  bucket: slemify-data
  path: k8s-autoscaling/data/
```

Key choices:
- **`task: generation`**: routes through the generative path — a causal LM
  downloaded, converted to GGUF, and quantized on CPU (no fine-tuning), then
  served on CPU and grounded by RAG. Unlike triage, the auditor must *write* a
  report, which only a generative model can do.
- **8B model**: large enough for structured reasoning with YAML output
- **`output_format: free_form`**: the auditor generates paragraphs, not labels
- **no synthetic data or labels**: the auditor isn't trained; its knowledge comes from RAG at serving time

## Step 4: Train and deploy

```bash
# Build both models (triage trains on CPU in seconds; the auditor converts on CPU, ~6 min)
slemify deploy --config triage/expert.yaml
slemify deploy --config auditor/expert.yaml
```

Each command runs the full pipeline: data prep, CPU training (triage) or GGUF convert and quantization (auditor), deployment to your cluster, and a validation report.

Monitor progress:

```bash
slemify status k8s-autoscaling-triage
slemify status k8s-autoscaling-auditor
```

## Step 5: View the reports

```bash
slemify report --config triage/expert.yaml
slemify report --config auditor/expert.yaml
```

The HTML report shows:
- Accuracy per label (does the model classify correctly?)
- Confidence calibration (when it says "high confidence," is it right?)
- SLM vs LLM comparison (how does it compare to Sonnet on the same queries?)
- Latency benchmarks (TTFT, tokens/second, throughput)
- Cost projections (what does this cost at 1K, 10K, 100K queries/day?)

If accuracy is below your threshold, add more training queries for the weak categories and retrain.

## Step 6: Set up the demo

Once both models are deployed and validated, set up the multi-agent demo:

```bash
cd examples/k8s-autoscaling/demo

# Full setup: OpenSearch, knowledge base, orchestrator
ARM64_BUILD_HOST=my-graviton-host ./scripts/setup-demo.sh
```

This deploys:
- OpenSearch for vector search (RAG)
- Knowledge base indexed from Karpenter, KEDA, and EKS docs (~3900 chunks)
- Orchestrator pod that ties everything together

## Step 7: Run it

```bash
# Launch the demo (port-forward + browser + tmux dashboard)
./scripts/demo-terminal.sh
```

Open `http://localhost:8000` and paste a Kubernetes config. You'll see:
1. Triage classification (1.5s)
2. RAG retrieval from the knowledge base
3. Auditor analysis streaming token by token

The tmux dashboard shows all three pods processing in sequence, proving it's a multi-agent system running entirely on CPUs.

## What you can customize

| What | How |
|------|-----|
| Different domain | Change `project.domain` and `labels` in expert.yaml |
| More training data | Add .txt files to your data directory, re-run pipeline |
| Different base model | Change `model.base` (a causal LM for `task: generation`; a sentence-transformers encoder for `task: classification`) |
| Larger context | Increase `CHUNK_SIZE` in index-knowledge.py for longer RAG chunks |
| Different vector DB | Replace OpenSearch with any k-NN capable store |
| GPU inference | Use vLLM instead of llama.cpp (different project, same GGUF model) |

## Cost breakdown

| Item | Cost | Frequency |
|------|------|-----------|
| Synthetic data (Bedrock, triage only) | ~$15 | One-time |
| Model prep (CPU: triage train / auditor convert) | <$1 | One-time |
| Inference (CPU Spot) | ~$117/mo per replica | Ongoing |
| OpenSearch (CPU) | ~$50/mo | Ongoing |
| LLM fallback (Bedrock) | ~$0.008 per query | Only for low-confidence queries |

At 10,000 queries/day with 85% handled by the SLM: ~$170/mo total vs ~$3,000/mo for 100% LLM API.

## Next steps

- Read the [Serving deep dive](deep-dive/serving.md) for autoscaling, TTFT optimization, and startup patterns
- Read the [Data deep dive](deep-dive/data.md) for training data best practices
- Try the [K8s Autoscaling Triage example](../examples/k8s-autoscaling/triage/) for a classification task
- Add your own domain knowledge to the RAG index
