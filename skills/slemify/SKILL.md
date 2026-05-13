---
name: slemify
description: Build specialized SLM agents for multi-agent systems. Identify where small language models replace LLM API calls, design the agent's role, generate training data, fine-tune, and deploy on Kubernetes with CPU inference.
---

# Slemify: Build Specialized SLM Agents

Slemify creates small language models (SLMs) that serve as specialized agents in multi-agent systems. An SLM built with Slemify is not a cheap replacement for an LLM. It's a dedicated component with a defined role, defined inputs, and defined outputs that handles one task faster and more accurately than a general-purpose model.

## Why CPU for SLM Inference

Inference for small models is memory-bandwidth bound, not compute-bound. Every token requires reading the model weights from RAM. A quantized 3B model (1.8GB at Q4) reads 3x faster than a full-precision model (6GB) from the same memory bus. Current-generation CPUs with high memory bandwidth (Graviton4, AMD EPYC Turin, Intel Xeon 6) deliver strong price-performance for models under 8B parameters.

The economics shift at scale: ~$117/month per CPU replica with unlimited queries vs $0.01+ per query with an LLM API. At 10,000 queries/day, that's a fixed $117 vs a variable $3,000. The SLM cost scales with infrastructure (replicas), not with traffic (queries).

For workloads that need lower latency or higher concurrency than CPU provides, the same GGUF model runs on GPU with vLLM. The model artifact is portable. The infrastructure choice is separate from the training choice.

For detailed guidance on when CPU inference is appropriate and how to optimize it, see the [EKS Best Practices for AI/ML CPU Inference](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-cpu-inference.html).

## The Adoption Path (Strangler Fig Pattern)

Don't replace the LLM overnight. The approach is:
1. Start with an LLM API. It works. It's fast to build.
2. Identify hot spots: repetitive tasks running thousands of times/day with the same pattern.
3. Fine-tune an SLM for that specific hot spot. Validate it matches or beats the LLM on that task.
4. Deploy the SLM. Keep the LLM as fallback for low-confidence cases.
5. Repeat for the next hot spot.

This is the same strangler fig pattern used to break monoliths into microservices. You're breaking a monolithic LLM dependency into specialized agents, one hot spot at a time.

## When to Use This Skill

Use this skill when a user:
- Wants to reduce LLM API costs for repetitive tasks
- Is building a multi-agent system and needs specialized components
- Has an existing LLM workflow with identifiable hot spots
- Wants to run inference on CPU (no GPU required for serving)
- Needs to keep data in their VPC (no external API calls at inference time)

## Phase 1: Discover SLM Opportunities

Before writing any config, help the user identify WHERE an SLM fits in their system. Ask:

"Describe your current AI workflow. What LLM calls are you making today? How often? What does the input look like? What does the output look like?"

Look for these signals:

**Strong SLM candidates:**
- Same prompt template used with different inputs (thousands/day)
- Output is always one of N categories or a structured format
- The task is a decision point: route, classify, extract, validate
- An LLM gets domain-specific questions wrong (hallucination on internal APIs, proprietary configs, or recent documentation the LLM wasn't trained on)
- High volume makes per-token cost significant
- Data sovereignty requirements (inference must stay in your VPC)

**NOT good SLM candidates (keep using the LLM):**
- Every request is genuinely different (creative writing, open-ended Q&A)
- Output is free-form prose with no structure
- Low volume (under 100 requests/day, cost savings don't justify effort)
- No existing examples to seed from (need to collect data first)
- The task requires broad world knowledge, not domain-specific patterns

**Example discovery conversation:**

User: "We call Claude 10,000 times a day to classify support tickets into 7 categories, then route them to the right team."

Agent response: "That's a strong SLM candidate. It's high volume, structured output (7 fixed categories), and the same pattern every time with different ticket content. A 3B model fine-tuned on your real tickets would handle this at sub-second latency with no per-token cost. Want me to set this up?"

User: "We also use Claude to write personalized responses to each ticket."

Agent response: "That's not an SLM candidate. Each response is unique and requires creative language. Keep that on the LLM. But the classification step before it? That's where Slemify fits. The SLM classifies and routes, the LLM generates the response for the tickets that need one."

## Phase 2: Design the Agent's Role

Once you've identified the opportunity, define the agent precisely:

### Two Patterns

**Router Agent (1-4B model):**
- Makes a fast decision (classify, route, triage)
- Output: 2-10 tokens (a label, a category, a confidence score)
- No RAG needed (the decision is based on input patterns, not external knowledge)
- Sub-second latency, handles thousands of requests/day
- Example: "Is this query about billing, shipping, or technical support?"

**Analyst Agent (7-8B model):**
- Produces structured reasoning about a specific domain
- Output: 100-500 tokens (analysis with explanation, evidence, recommendations)
- Uses RAG to ground reasoning in current documentation
- Streams response over 10-14 seconds, provides depth not speed
- Example: "What's wrong with this Kubernetes config, why is it dangerous, and how to fix it?"

### Key Design Questions

Ask the user:
1. "What decision does this agent make?" (defines the task)
2. "What are the possible outputs?" (defines the labels/schema)
3. "What does the input look like?" (defines data format)
4. "Who calls this agent and what happens with the result?" (defines integration)
5. "Do you have real examples of this task being done correctly?" (defines data availability)
6. "Does the agent need access to current documentation at query time?" (determines if RAG is needed)

### The SLM's Place in the System

Always frame the SLM as one node in a larger system:

```
[Request] --> [Router SLM] --> high confidence --> [Analyst SLM + RAG] --> response
                           --> low confidence  --> [LLM API fallback]  --> response
                           --> noise           --> rejected
```

The LLM stays in the system. It handles the 10-15% the SLM isn't confident about. The architecture is triage, specialist, fallback. Not "replace the LLM entirely."

## Phase 3: Build with Slemify

### Prerequisites

Verify the user has:
- An EKS cluster with Karpenter installed
- An S3 bucket for data and model artifacts
- AWS credentials with Amazon Bedrock access (for synthetic data generation)
- `kubectl` configured for their cluster
- The `slemify` CLI installed

If the cluster isn't set up yet, guide them through:
```bash
# Slemify needs: EKS + Karpenter + KEDA + S3 CSI driver (optional but recommended)
# See: https://github.com/aws-samples/sample-slemify/blob/main/docs/getting-started.md
```

### Writing expert.yaml

Based on the design phase, create the config. The base model comes from HuggingFace Hub (any Unsloth-compatible architecture).

**Router Agent template:**

```yaml
apiVersion: slemify/v1

project:
  name: <descriptive-name>
  domain: >
    <One paragraph describing what this agent does, what it classifies,
    and what the categories mean. Be specific.>
  labels:
    <primary_dimension>:
      - category_1
      - category_2
      - category_3

model:
  base: ""  # HuggingFace model ID (3B recommended for classification)
  quantize: q4_k_m

data:
  bucket: <your-s3-bucket>
  path: <project-name>/data/
  sources:
    - path: examples/
      type: raw
  synthetic:
    model: <bedrock-model-id>         # e.g., eu.anthropic.claude-sonnet-4-6
    pairs: 800                        # 500-1000 for classification

training:
  spot: true
```

**Analyst Agent template:**

```yaml
apiVersion: slemify/v1

project:
  name: <descriptive-name>
  domain: >
    <One paragraph describing what this agent analyzes, what domain
    expertise it has, and what structured output it produces.>
  labels:
    <error_type_or_category>:
      - type_1
      - type_2
    <severity_or_dimension>:
      - high
      - medium
      - low

model:
  base: ""  # HuggingFace model ID (8B recommended for structured reasoning)
  quantize: q4_k_m

data:
  bucket: <your-s3-bucket>
  path: <project-name>/data/
  sources:
    - path: examples/
      type: raw
  synthetic:
    model: <bedrock-model-id>
    pairs: 2500                       # More pairs for reasoning tasks

training:
  spot: true
```

### Data Preparation

The user needs raw examples uploaded to S3:

```bash
# Upload your real examples (minimum 50, ideally 100+)
aws s3 sync ./my-examples s3://<bucket>/<project>/data/examples/
```

**What are "raw examples"?**
- For a router: real queries/tickets/requests that represent each category
- For an analyst: real inputs paired with what a correct analysis looks like
- Quality matters more than quantity. 50 well-chosen examples beat 500 noisy ones.

**If the user already has a formatted dataset** (e.g., from HuggingFace Hub or their own labeling), they can skip synthetic generation by providing pre-formatted JSONL directly. Slemify's synthetic data generation is the default path for users who have raw examples but not structured training pairs.

**How synthetic data generation works:**
Slemify calls Amazon Bedrock with your raw examples and expert.yaml config. The LLM reads your real examples to understand the patterns, then generates realistic variations with correct labels. The domain expertise comes from YOUR examples, not from the LLM's general knowledge. The LLM is a pattern amplifier, not a domain expert.

### Running the Pipeline

```bash
slemify deploy --config expert.yaml
```

This runs: data generation, training (Spot GPU, ~20 min), quantization, serving, and validation report.

Check status:
```bash
slemify status <project-name>
```

### What to Expect

| Stage | Duration | Cost |
|-------|----------|------|
| Data (synthetic generation) | 5-15 min | $10-50 (Bedrock API) |
| Training (QLoRA on Spot GPU) | 15-30 min | $0.15-0.50 |
| Quantization | 2-5 min | included |
| Serving + Validation | 5-10 min | ~$0.10 |
| **Total** | **30-60 min** | **$15-55** |

## Phase 4: Validate

```bash
slemify report --config expert.yaml
```

Opens an HTML report with:
- **Overall accuracy** (target: 90%+ for router, 80%+ for analyst)
- **Per-class accuracy** (identifies weak categories)
- **Latency** (p50, p95 for SLM vs LLM baseline)
- **Cost projections** (monthly cost at various request volumes)
- **Consistency** (same input produces same output across repeated calls)

### Interpreting Results

**Good to deploy:**
- Router: 90%+ accuracy, sub-second latency, 100% consistency
- Analyst: 80%+ accuracy (judged by LLM), 1-2s TTFT, streaming

**Needs work:**
- One class significantly lower than others: add more raw examples for that class
- Overall accuracy below threshold: check if labels are ambiguous, consider merging similar categories
- High latency: check mlock is enabled, verify model fits in RAM

**Not ready:**
- Below 75% accuracy: the task may not be suitable for an SLM, or the raw examples don't represent the real distribution

## Phase 5: Integrate into Multi-Agent System

The deployed SLM exposes an OpenAI-compatible API (`/v1/chat/completions`). Any orchestrator can call it via HTTP:

```python
import httpx

response = httpx.post("http://triage-inference:8080/v1/chat/completions", json={
    "model": "model",
    "messages": [{"role": "user", "content": user_query}],
    "max_tokens": 32,       # Short for router, longer for analyst
    "temperature": 0.0,     # Deterministic for classification
})
```

### Integration Patterns

**Router into Analyst into Fallback:**
```
User query --> Router SLM (classify + confidence)
  --> high confidence + category A --> Analyst SLM + RAG --> structured response
  --> high confidence + category B --> Different handler
  --> low confidence --> LLM API (Bedrock) + RAG --> response
  --> noise --> reject
```

**Router into Direct action:**
```
User query --> Router SLM (intent + entity extraction)
  --> "cancel_order" + order_id --> Call cancellation API
  --> "track_shipment" + tracking_id --> Call tracking API
  --> ambiguous --> LLM for clarification
```

### Scaling

The SLM deployment includes KEDA autoscaling by default:
- Scales on queue depth (leading indicator), not CPU utilization (lagging)
- New replicas ready in ~55 seconds (Karpenter node provisioning + SOCI parallel image pull + warmup with readiness gate)
- Each replica: ~$117/month on Graviton Spot, unlimited queries
- Stateless inference pods are ideal for Spot (60-70% savings, near-unlimited CPU Spot capacity)
- Model loaded via S3 Mountpoint CSI (mmap, no download step) with mlock to prevent memory degradation
- Multiple replicas provide availability during Spot reclamation (Karpenter provisions a replacement node while remaining replicas continue serving)

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Low accuracy on one class | Not enough examples for that class | Add 10-20 more raw examples, regenerate |
| Model outputs extra text beyond label | Training data has inconsistent format | Check synthetic pairs, ensure clean pipe-delimited output |
| High latency (>3s for router) | mlock not enabled, model paging to disk | Enable mlock in deployment, verify RAM > model size |
| Training OOM | Model too large for GPU | Use Spot g5.xlarge (24GB) or reduce batch size |
| Spot interruption during training | Normal | Slemify checkpoints to S3, resumes automatically |
| Report shows "not production ready" | Accuracy below threshold | Review per-class breakdown, add examples for weak classes |

## Limitations

Be transparent with users about what Slemify does NOT do:
- Does not handle preference alignment (DPO/RLHF). Use TRL directly for that.
- Does not work for open-ended generation tasks. Keep those on the LLM.
- Synthetic data generation assumes the LLM can generate realistic variations from your examples. If your domain is extremely proprietary with no public documentation, review the generated pairs carefully.
- Currently deploys on Amazon EKS only. The GGUF model file is portable to any llama.cpp-compatible runtime.
- CPU inference has a throughput ceiling determined by memory bandwidth. For high-concurrency, long-output workloads, consider GPU serving with vLLM.

## Resources

- Repository: https://github.com/aws-samples/sample-slemify
- Getting Started: https://github.com/aws-samples/sample-slemify/blob/main/docs/getting-started.md
- EKS Best Practices for CPU Inference: https://docs.aws.amazon.com/eks/latest/best-practices/aiml-cpu-inference.html
- Base models: Any Unsloth-compatible model from HuggingFace Hub
