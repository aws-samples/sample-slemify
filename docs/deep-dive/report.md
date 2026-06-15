# Report Stage

The report stage runs your fine-tuned model against evaluation data and presents the results so you can decide if it's ready for your use case. It does not tell you whether the model is "good" or "bad." It shows you what the model does, how confident it is, how it compares to an LLM API, and what it would cost to run.

The metric is chosen by `project.task`:

- **Generation** (`task: generation`) → the LLM-as-judge report described in the rest of this document (semantic correctness, confidence from logprobs, LLM baseline, cost projections).
- **Classification** (`task: classification`) → exact-match accuracy plus per-class precision/recall/F1, computed by the training job and printed at the end of serving. No LLM judge — the labels are a closed set, so exact match is the honest metric.
- **Scoring** (`task: scoring`) → regression metrics: MAE (mean absolute error), RMSE, R², and the correlation between predicted and true scores. Lower MAE is better; R² near 1.0 and high correlation mean the head tracks the rubric. These are computed on the held-out eval set by the training job.
- **Embedding** (`task: embedding`) → retrieval metrics: recall@1/5/10 and MRR, reported as **stock vs tuned** so the fine-tuning gain is explicit. Each eval query's gold document must be retrieved from the corpus; higher recall@1 means the right document ranks first more often. Computed by the training job on the held-out query set.
- **Reranking** (`task: reranking`) → ranking metrics over a hard candidate list (NDCG@5, recall@1/5, MRR) for the **stock cross-encoder served on CPU**. Reranking is serve-only (no fine-tune): a strong general-purpose cross-encoder is already well-calibrated, and fine-tuning on synthetic single-positive data degrades it (mined negatives are often false negatives). The report shows the served model's ranking quality rather than a tuned delta.

The rest of this document covers the generation path's report in detail.

## What the report does

```
eval.jsonl samples
        │
        ├──► SLM inference (your model, via live endpoint)
        │         → predictions, latency, confidence from logprobs
        │
        ├──► LLM-as-judge (Bedrock)
        │         → semantic correctness + reasoning per prediction
        │
        ├──► LLM baseline (Bedrock zero-shot, same samples)
        │         → latency comparison, output format comparison
        │
        └──► Spot/On-Demand pricing (EC2 API)
                  → cost projections across CPU, GPU, and LLM API
```

1. **SLM inference.** Each eval sample is sent to the live inference endpoint. The response includes the prediction, latency, and token-level confidence from logprobs.
2. **LLM-as-judge.** A foundation model judges each prediction semantically. Not by string matching, but by understanding whether the model identified the correct category. The judge provides a one-line reasoning for each verdict.
3. **LLM baseline.** The same samples are sent to a Bedrock LLM (zero-shot, no fine-tuning). This shows the latency and output format difference. The LLM typically gets the intent right but outputs verbose prose instead of structured labels.
4. **Cost projections.** Real pricing from the EC2 API, showing what it would cost to run the SLM at different traffic volumes on CPU (on-demand and Spot), GPU, and compared to LLM API pay-per-request pricing.

## Why LLM-as-judge instead of string matching

Early versions of the report used deterministic string comparison to score predictions. This broke constantly because fine-tuned models produce output that is semantically correct but not string-identical to the expected label. For example:

- Model outputs `high|spot_interruption_handling` when expected is `high|spot_interruption`. Correct routing, more specific label.
- Model outputs `high|karpenter_config|medium|pdb_disruption` for a query about both topics. Correct multi-label classification.
- Model outputs `high` confidence when expected says `medium`. Debatable, but the routing is right.

Deterministic scoring marks all of these as failures. An LLM judge understands that they are correct. The judge prompt instructs it to focus on whether the primary category is right, accepting secondary labels, confidence disagreements, and more specific wording as valid.

This approach costs ~$0.15 per report (100 judge calls) and eliminates the need for fragile parsing logic that breaks every time the model produces slightly different output.

## Model confidence from logprobs

Every prediction includes a confidence score derived from the model's token probabilities (logprobs). This is not self-reported confidence. It is the actual probability the model assigned to each token in its output.

The confidence is computed as the geometric mean probability of the classification tokens (first line of output, excluding any thinking tokens). A 93% confidence means the model was very certain. A 51% confidence means it was nearly guessing between two options.

This matters because:

- High confidence + correct = the model knows this category well
- Low confidence + correct = the model got lucky or the input is ambiguous
- High confidence + incorrect = interesting failure mode worth investigating
- Low confidence + incorrect = expected, the model was uncertain and got it wrong

The confidence score is more informative than a binary correct/incorrect because it tells you where the model's decision boundaries are weak.

## Reading the report

The report is a self-contained HTML file with four tabs.

### Top metrics

Always visible regardless of which tab is active:

- Judge accuracy: percentage of predictions the LLM judge marked as semantically correct
- Avg model confidence: mean confidence from logprobs across all predictions
- SLM latency (p50): median end-to-end response time
- LLM latency (p50): median response time from the LLM API baseline
- Cost per node/mo: monthly cost of one inference node (CPU Spot)

### Predictions tab

The full table of every eval sample with: input, expected output, model's prediction, confidence percentage, judge verdict, judge's reasoning, and latency.

Look for:

- Predictions marked incorrect. Read the judge's reasoning to understand why.
- Low confidence predictions. These are where the model is uncertain.
- Patterns in failures. Are they all from one category? One type of input?

### SLM vs LLM tab

Latency comparison table showing:

- End-to-end latency: p50, p95, p99, min, max
- TTFT (Time to First Token): how long before the model starts generating
- ITL (Inter-Token Latency): time between each generated token
- Generation throughput: tokens per second
- Speedup factor: how much faster the SLM is vs the LLM API

Below the latency table, a side-by-side comparison of predictions. The SLM produces structured pipe-delimited output. The LLM produces varying formats (markdown, prose, JSON). Both get the intent right. The SLM just speaks the right language for downstream systems.

### Cost tab

Monthly cost projections at four traffic volumes (1K, 10K, 100K, 1M requests/day), comparing:

| Option | Characteristics |
|--------|----------------|
| CPU On-Demand | Predictable cost, no interruptions, moderate latency |
| CPU Spot | ~60% cheaper, risk of interruption (mitigated by PDB + multiple nodes) |
| GPU | Higher cost per node but 10-20x more throughput, fewer nodes needed at high volume |
| LLM API | Zero infrastructure, pay-per-request, cheapest at low volume, most expensive at high volume |

The table shows how many nodes are needed at each volume tier. At high volumes, GPU becomes cost-effective because one GPU node handles the throughput of 10-15 CPU nodes.

A note below the table covers the key decision factors: instance selection, capacity vs cost tradeoffs, fine-tuning investment, latency requirements, and operational complexity.

## What the report does not do

- No verdict. The report does not tell you "PRODUCTION READY" or "NOT READY." You know your domain's tolerance for misclassification, your latency requirements, and your budget. The report gives you the data to make that call.
- No consistency check. LLMs are non-deterministic by design. Running the same input multiple times and comparing outputs measures variation in secondary labels, not reliability. The judge accuracy already validates that the primary classification is correct.
- No accuracy threshold. A fixed "85% = good" threshold is meaningless without context. 75% accuracy might be fine if misroutes are handled gracefully downstream. 95% might not be enough if misclassification has high consequences.

## When accuracy looks low

If the judge marks many predictions as incorrect, investigate before assuming the model is bad:

1. Read the judge's reasoning. Is it flagging genuine misroutes (Karpenter question classified as noise) or debatable cases (confidence level disagreements)?
2. Check confidence scores. If the incorrect predictions all have low confidence, the model knows it is uncertain. The training data may not cover those cases well.
3. Check the eval data. If the eval data itself has questionable labels, the judge may be correctly identifying that the model's answer is better than the expected label.
4. Look at the training loss curve. If loss is still dropping at the end, more epochs might help. If it spiked at the end, the model may have overfit.

The fix is almost always in the data, not the model architecture or hyperparameters.

## Viewing the report

The report is uploaded to S3 at `{project}/report/report.html` during the pipeline run.

```bash
# Run the serving stage (deploys model + generates report)
slemify deploy --config expert.yaml --stage serving

# The report URL is printed at the end
# Download and open locally:
aws s3 cp s3://<bucket>/<project>/report/report.html report.html
open report.html
```

The HTML file is self-contained (no external dependencies) and includes a light/dark theme toggle.

## References

- [LLM-as-a-Judge on Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/llm-as-a-judge-on-amazon-bedrock-model-evaluation/). The approach Slemify uses for semantic scoring, using a foundation model to judge another model's output against a rubric.
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (NVIDIA, 2025). SLMs as high-frequency task handlers in multi-agent systems. The report validates whether your SLM is ready for that role.
- [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). The economics of inference at scale, why CPU serving makes sense for small models and when GPU becomes the better choice.
