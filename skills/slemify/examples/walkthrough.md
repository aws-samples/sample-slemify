# Walkthrough: Building a Support Ticket Router

This example shows the full flow of building a **classification** agent that
routes customer support emails into intent categories. It's a
`task: classification` model — a frozen text encoder plus a small trained head —
that trains and serves entirely on CPU (no GPU, no GGUF).

## The Scenario

A company receives 5,000 support emails per day. Today they call an LLM API for
each one to classify the intent and route it to the right team. Cost:
~$1,500/month. They want to cut that while keeping accuracy.

## Step 1: Discovery

A strong SLM candidate:
- High volume (5,000/day)
- Structured output (7 fixed categories)
- Same pattern every time (email in, category out)
- A decision, not prose — no creative generation needed

## Step 2: Design

- **Task:** `classification` (encoder + head, CPU)
- **Input:** raw customer email text (noisy: typos, mixed languages, OCR artifacts)
- **Output:** one intent category + a confidence (e.g., `billing_question|high`)
- **Categories:** billing_question, technical_issue, shipping_inquiry,
  refund_request, account_change, setup_help, feedback
- **Integration:** the orchestrator calls this first and routes on the intent.

Classification is single-label. If you also need another dimension (say,
sentiment), train a second classifier or use a `scoring` model — keep each head
focused on one decision.

## Step 3: Prepare Raw Examples

Collect ~55 real emails (at least 5 per category). Save as individual text files:

```
data/emails/billing-01.txt
data/emails/technical-01.txt
data/emails/shipping-01.txt
...
```

Each file is a raw email exactly as received (typos, formatting issues, etc.).

Upload to S3:
```bash
aws s3 sync ./data/emails s3://my-bucket/support-router/data/emails/
```

## Step 4: Write expert.yaml

```yaml
apiVersion: slemify/v1

project:
  name: support-intent-router
  task: classification
  domain: >
    Email triage for customer support. Classify a noisy, unstructured email
    into exactly one intent category. Inputs contain OCR artifacts,
    mobile-device typos, conversational tangents, and corrupted character
    encodings.
  labels:
    intent:
      - refund_request
      - setup_help
      - billing_question
      - technical_issue
      - feedback
      - account_change
      - shipping_inquiry

model:
  base: ""        # text encoder ID, 768d (trained + served on CPU; no GGUF)
  head: logistic  # logistic | linear | mlp

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

Output (encoder-head path — CPU, no GPU, no quantization):
```
Pipeline starting...
  [DATA]     generating 800 synthetic pairs from 55 raw examples...
  [TRAINING] encoder-head fit on CPU (frozen encoder + logistic head)...
  [QUANTIZE] n/a (encoder-head model)
  [SERVING]  deploying ONNX on Graviton CPU...
  [REPORT]   accuracy 95% (95/100), ~25ms/query

Done in ~12 minutes (dominated by synthetic data generation). Cost: ~$22.
Report: s3://my-bucket/support-router/report/report.html
```

## Step 6: Validate

```bash
slemify report --config expert.yaml
```

The report shows the encoder-head metrics:
- **Overall accuracy: 95%** (95/100 exact match)
- **Per-class:** billing 100%, technical 88%, shipping 92%, feedback 100%,
  setup 100%, refund 100%, account 82%
- **Latency: ~25ms/query** (encoder forward pass on CPU) vs ~1,700ms for the LLM API
- **Cost at 5,000 req/day: ~$117/month** (one CPU replica) vs ~$1,500/month (LLM API)
- **Consistency: 100%** (same input always produces the same label)

Assessment: production ready. The `account_change` class (82%) could improve with
more examples but is acceptable.

## Step 7: Integrate

The classifier serves an OpenAI-compatible endpoint and returns
`<label>|<confidence>` as the message content — a drop-in for a generative router.

```python
import httpx

result = httpx.post(
    "http://support-intent-router-inference:8080/v1/chat/completions",
    json={"model": "model", "messages": [{"role": "user", "content": email_text}]},
).json()

# Encoder-head classifier returns "intent|confidence", e.g. "billing_question|high"
intent, confidence = result["choices"][0]["message"]["content"].split("|")
# (Structured form is also under result["slemify"]: {"label", "probability"}.)

if intent == "billing_question":
    forward_to_billing_team(email_text)
elif intent == "technical_issue":
    forward_to_engineering(email_text)
# ... route the rest; send low-confidence cases to an LLM fallback
```

## Result

- Cost reduced from ~$1,500/month to ~$117/month (~92% savings)
- Latency dropped from ~1,700ms to ~25ms — the encoder runs entirely on CPU
- Accuracy maintained at 95% with domain-specific training
- No GPU and no per-token cost; data stays in the VPC (no external call at inference)
- Scales automatically with KEDA when email volume spikes
