# Report Stage

The report stage measures the model you just deployed and shows you the numbers you need to decide whether it is ready for the seat you want it in. It does not tell you whether the model is "good". It tells you how the model does on held-out data, how that compares to the simplest possible baseline, and what the served endpoint actually delivers in latency.

The report runs as a Kubernetes Job at the end of every `slemify deploy`, for every task family. It reads the held-out set and the training metrics from S3, calls the live inference endpoint, and writes two files to `s3://<bucket>/<project>/report/`:

- `report.json`, the data. `slemify deploy` and `slemify report` print a summary of it in the terminal.
- `report.html`, the same data as a self-contained page with the per-sample tables.

```
models/<project>/metrics.json ──┐
models/<project>/eval_predictions.jsonl ──┤
<project>/processed/eval.jsonl ──┤
                                 ├──► report Job ──► <project>/report/report.json
live inference endpoint ─────────┤                   <project>/report/report.html
node instance type + price ──────┘
   (resolved by the CLI)
```

## What each task family reports

The metric is chosen by `project.task`. The training job computes the quality numbers on the held-out set; the report Job adds the baseline, the split, the endpoint latency, and the optional frontier-model comparisons.

### Classification and extraction

- **Accuracy** (exact match) with the correct and total counts. The labels are a closed set, so exact match is the honest metric.
- **Majority-class baseline.** What you would get by always predicting the most common label in the held-out set. If the head is within 15 points of this number, the labels are probably not separable from the text, or a class has too few examples.
- **Real vs synthetic.** Accuracy on human-labeled records and on generated records, side by side. See [Human-labeled held-out data](#human-labeled-held-out-data) below.
- **Per-class precision, recall, F1** and the **confusion pairs** (expected label, predicted label, count), most frequent first. The confusions tell you which two labels the head cannot tell apart.
- **Calibration.** Predictions bucketed by the probability the head assigned, with the accuracy inside each bucket. A bucket that claims 90% and is right 60% of the time is where you should hand-check inputs. The classifier returns this probability on every request (`slemify.probability`), so you can use the same threshold for escalation in production.
- **Endpoint latency.** p50, p95, and max over a sample of held-out inputs sent to the served model, one request at a time. This is the number a caller sees, including the HTTP round trip inside the cluster.
- **Frontier-model zero-shot baseline** (optional, `report.llm_baseline: true`). The same held-out inputs sent to the Bedrock model configured for synthetic data, with the label set in the prompt, no examples. It answers "what would the frontier model get on this exact set" and its misses show which classes are ambiguous by definition. Off by default because it costs one Bedrock call per sample, at one request per second.

### Scoring

MAE, RMSE, R squared, and correlation between predicted and true scores, against the baseline of always predicting the mean. Lower MAE is better; R squared near 1.0 means the head tracks the rubric. Endpoint latency is measured the same way as for classification.

### Embedding

- **recall@1, @2, @5, @10 and MRR**, reported as **stock vs tuned**, so the gain from fine-tuning is explicit. recall@2 is the number to watch when the model feeds a reranker or a short context window: it says how often the right chunk is in the top two.
- **Real vs synthetic** recall for the tuned model, when human-written queries are provided.
- **Per-query ranks** in the HTML: for every held-out query, the rank of its gold chunk under the stock and the tuned model. The queries where the tuned model ranks the gold chunk worse than stock are the ones to read.
- **Endpoint latency** of the `/embed` route.

(Reranking is not a Slemify task; the training deep dive explains why fine-tuning a strong cross-encoder on synthetic data degrades it.)

### Generation

Slemify does not fine-tune generation models, so there is no held-out accuracy to report. What matters for a served generation model on CPU is whether it fits the latency budget of its seat, and that is what the report measures.

- **Model.** File, size on disk, parameter count, and training context, read from the server's `/v1/models` and `/props`.
- **Prefill and decode.** Time to first token cold (prompt not in the KV cache) and warm (same prefix, second call), prompt tokens per second, and decode tokens per second, from the server's own `timings` on two timed completions.
- **Bandwidth ceiling.** Decode speed on CPU is bounded by memory bandwidth divided by the bytes read per token, which for a dense model is the size of the weights. The report looks up the published per-socket bandwidth of the instance family (Graviton3, Graviton4, Graviton5, AMD Genoa, Intel Sapphire and Emerald Rapids), scales it by the share of the socket the node has, and states the ceiling next to the measured decode speed. The table is labeled as an estimate. If measured decode is far below the ceiling, the runtime or thread count is the problem; if it is at the ceiling, only a newer generation or a smaller model (fewer bytes per token, for example a mixture-of-experts) moves it.
- **Grounded evaluation** (optional, `report.cases`). A JSONL file of cases, each with a `question`, the `context` chunks the orchestrator would retrieve, and the `must_include` points a correct answer makes. The report drafts each case against the served model `report.repeat` times (default 2) and asks the Bedrock model whether each draft makes every required point. The result is a pass rate per case, not a single verdict, so a model that flips between runs shows up as 1/2 rather than as a coin toss. Off unless `cases` is set, because it costs Bedrock calls.

## Human-labeled held-out data

Generated evaluation data has the same blind spots as generated training data. A head that scores 98% on synthetic held-out records and 60% on questions users actually typed is a common outcome, and the report can only show it if you provide the human-labeled records.

Point `data.evaluation.labeled` at one or more JSONL files under `data.path`:

```yaml
data:
  evaluation:
    model: eu.anthropic.claude-sonnet-4-5-20250929-v1:0
    pairs: 150
    labeled:
      - path: eval-labeled/held-out.jsonl
```

Record shapes:

- classification, scoring, extraction: `{"input": "...", "output": "label"}`
- embedding: `{"query": "...", "positive": "chunk text"}` (the positive must appear in the corpus)

The data stage appends these records to `eval.jsonl` with `origin: real`; generated records carry `origin: synthetic`. Training metrics and the report score the two groups separately. A gap of more than 5 points between them is the first item in the report's guidance: the synthetic distribution has drifted from what users write, and the fix is the domain description and real seed examples, not the model.

Keep the labeled set out of the training sources. If a labeled record also appears as a seed, it is no longer held out.

## Serving cost

The report states one figure: the on-demand hourly rate of the instance type the inference pod landed on, for one node, looked up from the AWS Price List API by the CLI at report time (the Job itself has no pricing permissions). A node is fixed capacity, so the report does not turn this into a cost per request; that needs a request rate and a utilization target only you know. Earlier versions projected Spot, GPU, and API costs at several traffic tiers from fixed multipliers. Those numbers were not measured and have been removed.

## Configuration

```yaml
report:
  llm_baseline: true          # classification: frontier zero-shot on the held-out set
  cases: eval-cases/analyst.jsonl   # generation: grounded cases under data.path
  repeat: 3                   # generation: drafts per case (default 2)
  model: eu.anthropic.claude-sonnet-4-6   # Bedrock model for the two above; defaults to data.synthetic.model
```

All three are optional. Without them the report costs nothing beyond the Job's own CPU time and the requests it sends to your endpoint.

## Reading the terminal summary

```
  ━━━ Report: k8s-autoscaling-triage (classification) ━━━
  Accuracy:    97.3% (146/150) exact-match on the held-out set
  Baseline:    28.7% always predicting 'karpenter_config'
  By origin:   real 88.4% (43), synthetic 100.0% (107)
  Confusions:  hpa_config -> keda_config (3); multi_resource -> karpenter_config (1)
  Latency:     p50 9 ms, p95 14 ms, measured at the endpoint (50 requests)
  Node:        c8g.2xlarge (8 vCPUs), on-demand $0.3547/hour for one node
  Check first: Accuracy on human-labeled held-out data is 12 points below the synthetic set. ...
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Read it top to bottom. Accuracy against the baseline says whether the head learned anything. The origin split says whether what it learned matches real traffic. The confusions say where. Latency and node say what it costs to serve.

## What the report does not do

- No verdict. The report does not say "ready" or "not ready". You know your domain's tolerance for a wrong label, your latency budget, and what a frontier-model fallback costs you.
- No fixed threshold. 75% accuracy can be fine when the low-confidence cases escalate to a stronger model. 95% can be too low when a wrong label has real consequences.
- No frontier-model latency comparison. Comparing a CPU endpoint inside the cluster with an API call across the network measures the network, not the models.

## When a number is low

The report ends with the order to investigate, and it is the same for every task: the data first (coverage and labels), then the held-out set itself (are the expected labels right?), then the prompt or head settings, and only then a different model. For generation, most grounded-evaluation misses are the right chunk not being in the context, so check retrieval before the model.

## Viewing the report

```bash
slemify deploy --config expert.yaml          # runs the report at the end and prints the summary
slemify report --config expert.yaml          # prints the summary again and opens report.html
slemify report --config expert.yaml --no-open --output triage.html
```

The HTML file is self-contained and follows the system light or dark theme.

## References

- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (NVIDIA, 2025). SLMs as high-frequency task handlers in multi-agent systems. The report is how you check that a given SLM is ready for that role.
- [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). Why decode speed on CPU is a memory-bandwidth problem, which is what the generation report's ceiling arithmetic is based on.
