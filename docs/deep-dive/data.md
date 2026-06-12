# Data Stage

This is the stage where your effort matters most.

Training, serving, and reporting are largely automated. Slemify handles the infrastructure, the hyperparameters, and the deployment. But the data stage is different. The quality of the raw data you provide determines the ceiling of everything that follows. No amount of training tricks or infrastructure tuning will fix a model trained on bad data.

The philosophy is simple: **you bring the domain expertise, Slemify handles the automation.** You know your data better than any framework does. Your job is to provide diverse, representative raw examples and define the label taxonomy. Slemify takes that and generates the synthetic training pairs, validates them, checks for imbalances, and writes the final datasets to S3.

## How the pipeline works

```
Raw files in S3 (emails, logs, docs, tickets)    ◄── You provide these
        │
        ▼
Synthetic generation (Bedrock)  ◄── project.labels (strict enum)
        │
        ▼
Validation (pipe-delimited check)
        │
        ▼
Imbalance detection (warn if underrepresented)
        │
        ▼
train.jsonl + eval.jsonl + output_stats.json → S3
        │
        ▼
 ★ You verify the output ★                       ◄── Before training
```

1. **Read.** The pipeline reads raw source files from your S3 bucket. These are real examples from your domain, whatever your model will classify in production.
2. **Generate.** A large language model (Bedrock) reads your raw data and generates structured training pairs: an input (realistic text) paired with the correct output label.
3. **Validate.** Each generated record is checked for format compliance. Records that don't match the expected pipe-delimited output format are dropped.
4. **Balance check.** The pipeline logs the distribution of labels and warns when any class is underrepresented.
5. **Write.** Training and evaluation datasets are written as JSONL files to S3. The pipeline also writes `output_stats.json` with token length statistics (max, avg, p95) from the training outputs. The serving stage uses these stats to configure `max_tokens` and context window size for inference.

## Raw data is the single biggest factor

The quality of your raw source data matters more than the prompt, the model size, or the training hyperparameters. Everything downstream amplifies what's in the source data, both the signal and the bias.

The synthetic generator (Bedrock) uses your raw files as style and content references. If your source data only covers one type of input, the generator will produce variations of that one type. No amount of prompt engineering compensates for missing coverage in the source data.

**The practical rule:** your raw source data must cover every label combination you want the model to handle. If you have 3 categories, you need examples of all 3. If you have 20, you need examples of all 20. The number of categories doesn't matter. The coverage does.

<details>
<summary>Why synthetic data instead of manual labeling?</summary>

Manual labeling is accurate but expensive and slow. A domain expert would need to read and label hundreds of examples. Synthetic generation via a large language model produces labeled pairs at scale, generating hundreds of training pairs from a few dozen raw files in minutes.

The key insight is that the LLM isn't inventing knowledge. It's reading your real examples and generating realistic variations with correct labels. The raw data provides the "DNA": the specific patterns of noise, jargon, and structure that exist in your domain. The LLM stretches that DNA into a full training set.

Research supports this approach. The [Phi-1](https://arxiv.org/abs/2306.11644) (Microsoft, 2023) and [Phi-2](https://arxiv.org/abs/2309.05463) (Microsoft, 2023) papers demonstrated that models trained on high-quality synthetic data can match or outperform models trained on much larger datasets of web-scraped content. The quality of the generation prompt and the diversity of the seed data matter more than raw volume.
</details>

### What "good" raw data looks like

Good raw data is diverse, representative, and noisy in the ways your production data will be noisy.

| Property | Good | Bad |
|----------|------|-----|
| Coverage | Examples for every label category | All examples from one category |
| Noise variety | Reflects real production input (typos, formatting issues, jargon) | Clean, well-formatted text only |
| Length variation | Short inputs mixed with long ones | All roughly the same length |
| Edge cases | Ambiguous inputs that could belong to multiple categories | Only clear-cut, obvious examples |

The type of noise depends on your domain. Customer support emails have OCR artifacts and mobile typos. System logs have truncated stack traces and inconsistent timestamps. Financial documents have regulatory jargon and mixed currencies. Match the noise in your training data to the noise your model will see in production.

### How much raw data do you need?

The right amount depends on two things: how many label categories you have, and how much variation exists within each category.

**Raw source files** are not the training data. They're the seed material the LLM uses to generate synthetic pairs. A good starting point is 5-10 raw examples per label category. For a task with 3 categories, that's 15-30 files. For a task with 10 categories, that's 50-100 files.

Starting with too few examples risks biased output. If all your examples of one label happen to share an unrelated trait, the model will learn that spurious correlation instead of the actual classification signal.

**Synthetic pairs** scale independently from raw data. The `synthetic.pairs` setting in your config controls how many training examples the LLM generates. More categories or more variation within categories generally benefit from more pairs:

| Task complexity | Suggested pairs | Rationale |
|----------------|----------------|-----------|
| Binary classification (2 labels) | 200-300 | Simple decision boundary, fewer examples needed |
| Multi-class (3-7 labels) | 500-800 | More categories need more examples to distinguish boundaries |
| Fine-grained (8-20 labels) | 800-1500 | Closely related categories need more contrast |
| Multi-dimensional (labels × attributes) | 800+ | Combinatorial coverage requires more volume |

These are starting points. If accuracy on a specific category is low after training, the fix is almost always more diverse raw source data for that category, not more synthetic pairs overall.

## Label taxonomy must be explicit

When generating synthetic data, the LLM will invent synonyms for your labels unless you explicitly constrain it. A domain description that says "e.g., critical, warning, info" treats those as examples, not the exhaustive list. The generator might produce `warning` in one batch and `caution` or `alert` in another. Three labels for the same category.

Slemify solves this with the `project.labels` field in `expert.yaml`:

```yaml
# Example: alert classification
project:
  labels:
    severity:
      - critical
      - warning
      - info
    action:
      - page
      - ticket
      - ignore
```

```yaml
# Example: document routing
project:
  labels:
    type:
      - invoice
      - contract
      - correspondence
      - compliance
```

The pipeline reads these labels and injects them into the generation prompt as a strict enum: "VALID OUTPUT LABELS (use ONLY these exact values)". This is the standard pattern for classification systems. Define the label set before training.

### Domain description affects label generation

The `project.domain` text is included in every generation prompt. If your domain description uses words that overlap with your label values, Bedrock may invent labels from the description instead of using the configured ones.

For example, a domain description that says "determines whether a message is signal or noise" will cause Bedrock to generate `signal` as a label. even if `signal` isn't in your `project.labels`. The fix is to use the exact label values in the domain description, or avoid language that could be interpreted as label names.

**Bad:** "Classify messages as important or unimportant"
**Good:** "Classify messages into one category: billing_question, technical_issue, ... or general_inquiry"

### Meta-labels need prompt guidance

Labels that represent content categories (intent, routing, error type) are driven by raw data. more source files for a category means more generated examples for that category. But labels that represent meta-properties (confidence, certainty, severity) are harder to control through raw data alone.

A raw file that's a vague, ambiguous question doesn't automatically generate `low` confidence training pairs. Bedrock classifies its own generated content with high confidence because it's confident in its own classification. The ambiguity in the source material doesn't transfer to the output label.

For meta-labels, the pipeline includes a distribution guidance line in the label format instruction: "Distribute values across all options in each field." This nudges Bedrock to use all values, including ones it wouldn't naturally choose. This guidance is generic and applies to all projects with structured multi-field labels.

<details>
<summary>What happens without explicit labels?</summary>

Without explicit label constraints, even the same model will produce inconsistent labels across batches. In practice, a single generation run can produce three different labels for the same concept. For example, `technical`, `technical_support`, and `technical_issue` all appearing in the same dataset.

Using a different model for evaluation makes this worse. Each model has its own interpretation of label boundaries, so even with the same prompt constraints, label mismatches can cause apparent accuracy to drop dramatically. Not because the model is wrong, but because the labels don't match.

Amazon Bedrock's [Structured Outputs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html) feature (2025) uses JSON Schema with enum fields for guaranteed label compliance. Slemify applies the same principle at the prompt level, which works across all model providers.
</details>

## Independent evaluation data

Training and evaluation data must be generated independently. If you split a single batch 90/10, you're testing memorization, not generalization. The eval samples are stylistically identical to the training samples because they came from the same generation run.

Slemify supports independent eval generation through the `evaluation` block in `expert.yaml`:

```yaml
data:
  sources:
    - path: training-data/    # Raw files for training generation
      type: raw
  synthetic:
    model: eu.anthropic.claude-sonnet-4-6
    pairs: 500
  evaluation:
    model: eu.anthropic.claude-sonnet-4-6
    pairs: 100
    sources:
      - path: eval-data/      # Different raw files for eval generation
        type: raw
```

The independence comes from **different source data**, not from using a different model. The eval source files should be deliberately different: harder edge cases, different writing styles, scenarios the training data doesn't cover directly.

**Why the same model for both:** Using a different model for eval causes label taxonomy mismatches despite prompt constraints. Each model has its own interpretation of label boundaries. The independence that matters is in the source material, not the generator.

If you don't configure an `evaluation` block, the pipeline falls back to a 90/10 split of the training data. This works for quick iteration but isn't a reliable measure of how the model will perform on unseen inputs.

<details>
<summary>What does "100% accuracy" actually mean?</summary>

If your model achieves 100% accuracy on evaluation data, it almost certainly means your eval data is too similar to your training data. Both were generated by the same model from the same source material. The eval is testing memorization, not generalization.

Realistic accuracy for a fine-tuned SLM on an independently generated eval set is 80-95%, depending on the task complexity. The remaining gap represents genuinely ambiguous cases, inputs where reasonable humans would disagree on the label. In production, the downstream system handles these. A misrouted request gets recognized and re-routed. The cost of a misroute is slightly longer resolution time, not a failure.
</details>

## Class balance: detect, don't fix

After generation, the pipeline checks the distribution of labels and warns when any class is underrepresented. It does not automatically rebalance.

**Why not auto-balance?** Forcing the generator to produce equal numbers per label is fragile and domain-specific. In some domains, imbalance is real. Critical alerts might be 2% of all alerts, and that's the actual production distribution. In others, you genuinely need equal representation. The right fix depends on your domain:

- **Imbalance reflects reality** → Accept it. The model should learn the real distribution.
- **Imbalance is an artifact of limited source data** → Add more raw source files for the weak categories.

The pipeline warns. You decide based on your domain knowledge.

## Output shape by task

The shape of the synthetic data the pipeline generates is determined by
`project.task` (and, for generation, by `project.output_format`).

### Label output (classification and other encoder-head tasks)

For `task: classification`, the pipeline generates `input → label` pairs, where
the label is one value from your `project.labels` taxonomy (e.g. `hpa_config`).
Validation drops records with empty output, and the label-balance check warns
about underrepresented classes.

The label is consumed by the classifier head, not by a language model, so there
is no "output format" to choose — the encoder embeds the input and the head
predicts the class. Confidence is the head's probability, not a generated token.

> Historically Slemify expressed classification as a generative model emitting a
> `pipe_delimited` string. That was replaced by the encoder-head classification
> path, which is faster, deterministic, and CPU-trained. The `pipe_delimited`
> output format has been retired.

### Free-form output (generation tasks)

For `task: generation` with `output_format: free_form`, the pipeline generates
`input → reasoning` pairs: structured explanations (identification, analysis,
correction, risk assessment) rather than labels.

```yaml
project:
  task: generation
  output_format: free_form
```

Validation checks for non-empty output instead of a label. Free-form is designed
for larger models (7B+) that serve as expert auditors in a tiered architecture.
The `project.labels` field still guides the output structure: labels become
section headings (e.g., "Error Type: deprecated_api, Severity: critical") rather
than the predicted value.

<details>
<summary>When output format matters vs. when it doesn't</summary>

If a generative SLM's output is consumed by **code** (a routing function, API
gateway, switch statement), format compliance is critical — use constrained
decoding ([llama.cpp grammars](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md),
Bedrock structured outputs) to enforce it. (For pure routing, prefer
`task: classification` — a classifier returns a clean label with no parsing.)

If the output is consumed by **another LLM** (an orchestrating agent), format
matters less. The orchestrator can parse JSON, markdown, or free text. What
matters is whether the model gets the answer right.

This distinction comes from the [Microsoft Multi-Agent Reference Architecture](https://www.microsoft.com/en-us/research/publication/multi-agent-reference-architecture/) (2025), which describes the "Semantic Router" pattern: a model outputs an intent label, and an LLM orchestrator routes based on it.
</details>

Use free-form when the downstream consumer needs to understand *why*, not just *what*. For example, a triage classifier outputs `billing_question|high`. An auditor outputs a paragraph explaining what's wrong with a configuration and how to fix it.

## The generation prompt

The synthetic generation prompt is the contract between your domain knowledge and the LLM's generation capability. Slemify's prompt includes:

1. **Domain context.** Your `project.domain` description, telling the LLM what kind of data to generate.
2. **Source samples.** A random subset of your raw files, used as style and content references.
3. **Valid labels.** The strict enum from `project.labels`, with explicit instructions to use only these values.
4. **Format instructions.** JSONL output with `instruction`, `input`, and `output` fields. Pipe-delimited output only.
5. **Diversity instructions.** "Vary scenarios, writing styles, noise levels, and personas."

The prompt asks for batches of 5 records at a time. Smaller batches reduce the risk of truncated responses (a common issue when generating structured data with LLMs) and increase diversity across batches because different random source samples are selected each time.

## Verify the synthetic data before training

This is the most important step in the entire pipeline. Before moving to the training stage, download the generated `train.jsonl` and `eval.jsonl` from S3 and inspect them.

```bash
aws s3 cp s3://your-bucket/your-project/processed/train.jsonl .
aws s3 cp s3://your-bucket/your-project/processed/eval.jsonl .
```

What to look for:

- **Label correctness.** Spot-check 20-30 records. Are the labels right? If the generator consistently mislabels a category, the model will learn that mistake.
- **Output format.** Every `output` field should be pipe-delimited (e.g., `critical|page`). Records with JSON, prose, or mixed formats will confuse the model during training.
- **Input diversity.** Scroll through the inputs. Do they look like realistic variations of your domain, or are they repetitive rewrites of the same scenario?
- **Noise coverage.** Are some inputs clean and some messy? If everything looks polished, the model won't handle real-world noise.
- **Label distribution.** Check the balance. The pipeline logs warnings, but you should verify that the distribution makes sense for your domain.

If something looks off, **don't adjust the generation prompt.** The fix is almost always in the raw source data:

- Labels wrong for a category → Add clearer raw examples for that category.
- Missing a noise profile → Add raw examples that exhibit that noise.
- One category underrepresented → Add more raw source files for it.

Slemify is designed so that you iterate on the raw data, not on the internals. Re-run the data stage after improving your source files and verify again. This loop (raw data → generate → verify → improve raw data) is where the real work happens.

## Noise and domain realism

Training inputs need to reflect the mess that production data contains. The type of noise depends entirely on your domain:

| Domain | Typical noise | What to include in raw data |
|--------|--------------|----------------------------|
| Customer support | OCR artifacts, mobile typos, mixed languages, rambling | Real emails with real formatting issues |
| System logs | Truncated traces, inconsistent timestamps, interleaved output | Actual log snippets from production |
| Financial docs | Regulatory jargon, mixed currencies, scanned PDFs | Real documents with formatting artifacts |
| Code review | Incomplete snippets, mixed languages, inline comments | Actual code fragments and review comments |

The generation prompt asks the LLM to vary noise levels across examples: some clean, some messy, some extremely noisy. A model trained only on clean text will fail on noisy production inputs. A model trained on diverse noise profiles learns to focus on the semantic signal rather than surface-level formatting.

## References

- [Phi-1: Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) (Microsoft, 2023). Demonstrates that high-quality synthetic data can match web-scale training data.
- [Phi-2: The Surprising Power of Small Language Models](https://arxiv.org/abs/2309.05463) (Microsoft, 2023). 2.7B model matching 25x larger models through data quality.
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (NVIDIA, 2025). SLMs under 10B parameters can handle 60-80% of agentic tasks; advocates for heterogeneous architectures.
- [Microsoft Multi-Agent Reference Architecture](https://www.microsoft.com/en-us/research/publication/multi-agent-reference-architecture/) (2025). SLM classifiers for initial routing in multi-agent systems.
- [Amazon Bedrock Structured Outputs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html). JSON Schema with enum fields for guaranteed label compliance.
- [xLAM-2-8B](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r) (Salesforce). 8B model that beat GPT-4o at tool calling, proving specialization beats size.
- [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). Why inference economics are driven by memory bandwidth, not compute.
