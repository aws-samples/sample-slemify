# Walkthrough: Building a Support Ticket Router

This example shows the full flow of building a Router Agent that classifies customer support emails into intent categories.

## The Scenario

A company receives 5,000 support emails per day. Currently, they call Claude for each one to classify the intent and route to the right team. Cost: ~$1,500/month. They want to reduce this while maintaining accuracy.

## Step 1: Discovery

The agent identifies this as a strong SLM candidate:
- High volume (5,000/day)
- Structured output (7 fixed categories)
- Same pattern every time (email in, category out)
- No creative generation needed

## Step 2: Design

**Agent role:** Router Agent (3B model)
**Input:** Raw customer email text (noisy, typos, mixed languages)
**Output:** `intent|sentiment` (e.g., `billing_question|frustrated`)
**Categories:** billing_question, technical_issue, shipping_inquiry, refund_request, account_change, setup_help, feedback
**Sentiment:** angry, frustrated, neutral, satisfied
**Integration:** Orchestrator calls this first, routes to appropriate handler based on intent

## Step 3: Prepare Raw Examples

Collect 55 real emails (at least 5 per category). Save as individual text files:

```
data/emails/billing-01.txt
data/emails/billing-02.txt
data/emails/technical-01.txt
data/emails/shipping-01.txt
...
```

Each file is a raw email exactly as received (with typos, formatting issues, etc.).

Upload to S3:
```bash
aws s3 sync ./data/emails s3://my-bucket/support-router/data/emails/
```

## Step 4: Write expert.yaml

```yaml
apiVersion: slemify/v1

project:
  name: support-intent-router
  domain: >
    Email triage for customer support. Extract intent and sentiment
    from noisy, unstructured emails containing OCR artifacts,
    mobile-device typos, conversational tangents, and corrupted
    character encodings.
  labels:
    intent:
      - refund_request
      - setup_help
      - billing_question
      - technical_issue
      - feedback
      - account_change
      - shipping_inquiry
    sentiment:
      - angry
      - frustrated
      - neutral
      - satisfied

model:
  base: ""  # HuggingFace model ID (3B recommended for classification)
  quantize: q4_k_m

data:
  bucket: my-bucket
  path: support-router/data/
  sources:
    - path: emails/
      type: raw
  synthetic:
    model: eu.anthropic.claude-sonnet-4-6
    pairs: 800
  evaluation:
    model: eu.anthropic.claude-sonnet-4-6
    pairs: 100

training:
  spot: true
```

## Step 5: Run

```bash
slemify deploy --config expert.yaml
```

Output:
```
Setting up Pod Identity...
Verifying S3 bucket encryption...
  S3 bucket encryption verified

Setting up Karpenter NodePools...
  slemify-slm NodePool applied
  slemify-gpu NodePool applied

Pipeline starting...
  [1/5] Data: generating 800 synthetic pairs from 55 raw examples...
  [2/5] Training: QLoRA on Spot GPU (3B model)...
  [3/5] Quantize: GGUF Q4_K_M...
  [4/5] Serving: deploying on Graviton CPU...
  [5/5] Report: evaluating 100 samples...

Done in 38 minutes. Total cost: ~$22.
Report: s3://my-bucket/support-router/report/report.html
```

## Step 6: Validate

```bash
slemify report --config expert.yaml
```

Report shows:
- **Overall accuracy: 95%** (95/100 correct)
- **Per-class:** billing 100%, technical 88%, shipping 92%, feedback 100%, setup 100%, refund 100%, account 82%
- **Latency p50: 1,510ms** (SLM) vs 1,701ms (LLM baseline)
- **Cost at 5,000 req/day: $117/month** vs $1,500/month (LLM API)
- **Consistency: 100%** (same input always produces same output)

Assessment: Production ready. The account_change class (82%) could improve with more examples but is acceptable.

## Step 7: Integrate

The SLM is now serving at `http://support-intent-router-inference:8080` in the cluster.

```python
# In your orchestrator:
result = httpx.post(
    "http://support-intent-router-inference:8080/v1/chat/completions",
    json={
        "model": "model",
        "messages": [{"role": "user", "content": email_text}],
        "max_tokens": 10,
        "temperature": 0.0,
    }
).json()

# Parse: "billing_question|frustrated"
intent, sentiment = result["choices"][0]["message"]["content"].split("|")

# Route based on intent
if intent == "billing_question":
    forward_to_billing_team(email_text, sentiment)
elif intent == "technical_issue":
    forward_to_engineering(email_text, sentiment)
# ... etc
```

## Result

- Cost reduced from $1,500/month to $117/month (92% savings)
- Latency improved by 11% (SLM is faster than LLM API)
- Accuracy maintained at 95% (with domain-specific training)
- No external API calls at inference time (data stays in VPC)
- Scales automatically with KEDA when email volume spikes
