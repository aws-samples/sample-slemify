# Training Stage

The training stage is fully automated. Once you've verified your synthetic data (see [Data Stage](data.md)), Slemify handles model loading, training (for the encoder families), conversion and quantization (for generation), and artifact upload. You don't need to write training scripts, pick hyperparameters, or manage instances. Everything runs on CPU.

Your only decision at this stage is which base model to use and whether to enable Spot instances for the trained tasks. Everything else is auto-sized based on the model and dataset.

## How each task family is built

Slemify's "training" stage covers two real training paths plus one no-training
path, chosen by `project.task`. The **encoder-head** path (`classification`,
`scoring`) and the **embedding** path train on CPU and are described below. The
**generation** path does not fine-tune at all: it downloads a base model,
converts it to GGUF, and quantizes it. All three share the data and report
stages.

| | Generation (`task: generation`) | Encoder-head (`classification`, `scoring`) | Embedding (`task: embedding`) |
|---|---|---|---|
| What's trained | Nothing — served stock | A small head on a frozen encoder | The encoder itself, contrastively |
| Engine | llama.cpp convert + quantize | scikit-learn (logistic/ridge) | sentence-transformers (MultipleNegativesRankingLoss) |
| Hardware | **CPU** | **CPU** | **CPU** |
| Time | ~5-8 min (download + convert + quantize) | **seconds to a couple of minutes** | **a few minutes** |
| Output | GGUF (quantized) | `head.json` + `encoder.onnx` + `tokenizer.json` | `encoder.onnx` + `tokenizer.json` (no head) |
| Why | The base model already generates and reasons; the domain knowledge comes from RAG at serving time, not from training | The encoder already understands language; only a decision rule is learned | The encoder must learn *your domain's* notion of similarity |

The encoder-head path is fast because nothing in the billion-parameter encoder
is updated. The training job embeds each input once (a forward pass through the
frozen encoder, run in-process with sentence-transformers), then fits a small
head over those vectors: a logistic head (`embedding_dim × num_classes` weights)
for classification, or a ridge regression head (`embedding_dim` weights + an
intercept) for scoring. There is no backpropagation through the encoder, no
epochs, and no GPU. The same job then exports the frozen encoder to ONNX so
serving needs neither torch nor sentence-transformers. The expensive, GPU-hungry
work (learning language) was already done once when the encoder was pretrained;
every encoder-head task rides on top of it cheaply.

The **embedding** path is the one encoder family that *does* update the encoder.
It runs contrastive training: each `(query, positive)` pair is pulled together
in vector space while the other documents in the same batch act as in-batch
negatives (MultipleNegativesRankingLoss). This is still CPU-feasible for the
small encoders Slemify targets — sequence length is capped (256 tokens) and the
batch kept modest to bound memory. The job measures retrieval recall@k/MRR
before and after fine-tuning (stock vs tuned) so the gain is explicit, saves the
tuned encoder, and exports *that* to ONNX. A domain corpus typically lifts
recall@1 by 10+ points over a stock general-purpose encoder in a couple of
minutes of CPU training.

### Extraction: a feature-based token tagger (no encoder)

The **extraction** task is the one encoder-family member that, in v1, uses no
encoder at all. Token-level span extraction needs a *per-token* prediction, which
the pooled-embedding + sklearn-head recipe (one vector per input) cannot produce.
Rather than reach for a heavier token-classification fine-tune, Slemify ships a
classic, CPU-instant tagger: logistic regression over per-token features (the
token, its word shape, affixes, casing, and a ±2-word context window) predicting
BIO tags, decoded into typed spans. It trains in seconds, and serving reproduces
the linear model in numpy — no encoder, no ONNX, no torch.

We added extraction only after measuring that it *earns* training in the right
domain. On open-vocabulary prose (software support tickets) the tagger lifts
entity-level span F1 from 0.63 (a strong regex + gazetteer baseline) to 0.89 — the
gain concentrated entirely on the open-vocab entities (service and error names)
that a regex can't enumerate. The training report also prints a
*memorize-training-surfaces* baseline (string-match the entity surfaces seen in
training); the tagger beats that too, which is the real test — it confirms the
model generalizes to surfaces it never saw rather than memorizing a lookup table.

The flip side is documented honestly: on *structured* text (Kubernetes YAML
configs) a plain parser extracts `apiVersion`/`kind` perfectly, so a trained model
adds nothing. That is why the extraction example lives in a support-ticket domain,
not the k8s one. The same domain-vs-data-shape distinction below (for reranking)
applies: extraction fails only in the wrong *domain*, and its training data
(text with labeled spans) is synthesizable — so it gets a fair trial in a domain
where it helps. A heavier neural token-classification fine-tune is possible future
work if a domain needs a higher ceiling.

### Why reranking is not a Slemify task (a documented learning)

A reranker is a *cross-encoder*: it reads the query and document together (one
joint forward pass per pair) and emits a single relevance logit — more precise
than a bi-encoder's independent vectors, but too expensive to run over a whole
corpus, so it only re-orders a small candidate set from first-stage retrieval.

We built and tested reranker fine-tuning, then **removed it**, because it didn't
help — and Slemify's premise is to build specialist models *when training adds
value*. The findings, kept here as a learning:

- A strong general-purpose cross-encoder is *already*
  well-calibrated for (query, document) relevance. On a fair hard eval (rank the
  answer among its most confusable near-misses) the stock model scored NDCG@5
  0.85.
- Reliable cross-encoder fine-tuning needs **curated hard negatives** — documents
  that look relevant but are verified not to be. We can't synthesize those: our
  pipeline only knows one positive per query, and "negatives" mined from an
  overlapping technical corpus are frequently relevant themselves (false
  negatives). Training on those teaches the model to push down good documents.
- Result: fine-tuning *degraded* the model — NDCG@5 0.85 → 0.58, recall@1
  78.6% → 35.7%. Random negatives regressed it slightly; hard (lexical) negatives
  regressed it badly, because they were the most likely to be false negatives.
- Contrast with embedding: the bi-encoder *improved* (+11.9 pts recall@1) on the
  same data, because its contrastive in-batch-negative loss tolerates label noise.
  Cross-encoder relevance training does not.

Getting this data properly would mean human relevance annotation, click logs from
a live system, or an LLM-as-judge step that labels and *filters* mined negatives —
a real data project, not a byproduct of a doc corpus, with an uncertain payoff for
an already-strong base model. So Slemify doesn't fine-tune rerankers. Running a
stock cross-encoder reranker on CPU (no GPU) is still useful, and the
k8s-autoscaling demo shows exactly that as a standalone serving pattern.

**Is this just a wrong-domain result?** A fair question, since k8s is unusually
parser- and gazetteer-friendly. We re-examined it and the answer is no — the
reranking blocker is more fundamental than the domain. It helps to separate two
ways a fine-tune can fail to beat a strong baseline:

- **Domain gap (fixable).** The base model is weak on *your* data. Pick a domain
  where the general model struggles and tuning pays off. The bi-encoder retriever
  is this case: domain tuning sharpened its vector space (+11.9 pts recall@1).
- **Data shape (not fixable by domain choice).** The training signal the method
  needs isn't something you can produce from your inputs. Reranking is this case:
  it needs *graded relevance judgments* or *verified* hard negatives, which come
  from click logs or human raters — not from a document corpus. That gap follows
  reranking into every domain. Even for legal, patent, biomedical, or code search
  (where a general cross-encoder genuinely *is* weak and tuning *would* help), the
  enabler is a relevance-judgment dataset, which is exactly what Slemify's
  "synthesize training data from your docs" premise does not produce.

So reranking isn't excluded because k8s is a bad showcase; it's excluded because
the data it needs is structurally outside what Slemify synthesizes. (Extraction,
by contrast, fails *only* in the k8s domain — its data, text with labeled spans,
is synthesizable — so it gets a fair trial in a better-suited domain.)

**Future work, if revisited.** One method we did *not* test could fit Slemify's
premise: distilling a strong LLM's graded relevance scores (an LLM-as-judge over
many query–document pairs) into a small CPU cross-encoder, rather than mining
hard negatives. That would be its own gated experiment with its own honest
before/after — not a claim that it works, just the one avenue that stays inside
"synthesize the signal you need."

## Generation is served stock (no fine-tuning)

The generative path does not train the model. We tested fine-tuning the
generative auditor in the k8s-autoscaling example and it made answers *worse*:
for a knowledge task the base model already reasons and writes, and what it lacks
is your facts. RAG supplies those at serving time more reliably than baking them
into weights, and the answer's format is handled by prompting and constrained
decoding. So Slemify serves generation stock and spends its training effort on
retrieval (the `embedding` task) instead. See the repo FAQ "Does fine-tuning
always improve quality?" for the full rationale.

The stage is therefore a CPU convert-and-quantize job, not a training run:

```
base model (HuggingFace)
        │
        ▼
Download weights to a CPU node (on-demand)
        │
        ▼
Convert to GGUF (f16) with llama.cpp
        │
        ▼
Quantize (e.g. q8_0) with llama-quantize
        │
        ▼
Upload model-<quant>.gguf to S3
```

1. **Download.** The job pulls the base model's weights from HuggingFace to a CPU node.
2. **Convert.** llama.cpp's converter writes an f16 GGUF.
3. **Quantize.** `llama-quantize` produces the configured level (default `q4_k_m`; the k8s auditor uses `q8_0`, see below).
4. **Upload.** The quantized `model-<quant>.gguf` and conversion logs are uploaded to S3 for the serving stage.

There is no GPU, no adapter, and no checkpoint in this path. The job is a
one-shot, bandwidth-heavy run (an 8B model is ~16GB of weights to download and a
~16GB f16 GGUF to quantize), so Slemify pins it to **on-demand** capacity: a Spot
reclaim mid-run would force a full re-download. Everything else in the pipeline,
serving included, still runs on Spot.

## Choosing a base model

The base model is the one Slemify converts and serves. Slemify supports any HuggingFace causal LM that llama.cpp's GGUF converter can read. The choice depends on your task complexity.

For knowledge-grounded reasoning over your own corpus (the primary generation use case), pick the smallest model that answers your eval cases well once it is grounded by RAG. Smaller is cheaper and faster on CPU:

| Model size | Good for | Inference speed (CPU) |
|-----------|----------|----------------------|
| 1-3B | Simple structured answers, tight latency budgets | Fast. Sub-second to first token on most CPUs. |
| 7-8B | Multi-step reasoning, config analysis, structured generation | Moderate. 1-2s on CPU. |
| 13B+ | Complex reasoning, long-form generation | Slow on CPU. Consider GPU serving. |

**The practical guidance:** start with the smallest model that answers your eval cases well when grounded by RAG. You can always scale up if accuracy is too low, but you can't get back the latency and cost savings of a smaller model.

<details>
<summary>Why not always use the biggest model?</summary>

Bigger models are slower and more expensive to serve. On CPU (where Slemify deploys for inference), latency is roughly proportional to parameter count because inference is [memory-bandwidth bound](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). A 3B model moves about 1.8GB of weights through memory per token. An 8B model moves about 4.5GB. The CPU spends most of its time waiting for data, not doing math.

Research from [Microsoft](https://arxiv.org/abs/2309.05463) and [NVIDIA](https://arxiv.org/abs/2506.02153) confirms that models under 10B parameters match or beat larger models on structured, repetitive tasks, especially when grounded by retrieval for the domain knowledge.
</details>

## Quantization and GGUF export

The convert job exports the model to [GGUF format](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md) and quantizes it. This is what makes CPU inference possible: GGUF is the format llama.cpp serves, and quantization shrinks the model so each generated token streams fewer bytes through memory.

<details>
<summary>What is quantization?</summary>

A model's weights are normally stored as 16-bit floating point numbers (2 bytes each). Quantization reduces the precision to fewer bits per weight, shrinking the model and making it faster to load and read from memory.

For an 8B parameter model:
- **F16 (no quantization):** ~16GB. Full precision, slowest on CPU.
- **Q8_0 (8-bit):** ~8.7GB. Near-lossless.
- **Q5_K_M (5-bit):** ~5.5GB.
- **Q4_K_M (4-bit):** ~4.8GB. Smallest, fastest.

Why this matters for CPU inference: the bottleneck is [memory bandwidth](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/), not compute. Every token generated requires reading the model's weights from RAM, so a smaller model generates tokens proportionally faster.
</details>

Slemify supports these quantization levels:

| Level | Config value | Notes |
|-------|-------------|-------|
| 4-bit | `q4_k_m` (default) | Smallest and fastest. |
| 5-bit | `q5_k_m` | Middle ground. |
| 8-bit | `q8_0` | Near-lossless; the safe choice for reasoning-heavy tasks. |
| None | `f16` | Full precision; only useful for GPU serving. |

**Pick the quant against your eval, not by reputation.** The default `q4_k_m` is fine for many tasks, but quantization hurts *reasoning* far more than it hurts perplexity, and a binary pass/fail eval over calibration-heavy cases exposes that. The k8s-autoscaling auditor is a clear example: on the demo's end-to-end scorecard, q8_0 held **15/18**, while q5_k_m and q4_k_m collapsed to **7/18** and **8/18** by losing calibration (inventing problems on valid configs). So the auditor is served at `q8_0` even though it is larger and slower. Always re-run the [report](report.md) scorecard after changing the quant.

<details>
<summary>Can a smaller quant be made to hold accuracy?</summary>

Plain post-training quantization (what the convert job does) is the cheap path. If you need a smaller model that still holds accuracy on a reasoning task, the recovery options, cheapest first, are: importance-matrix (imatrix) calibrated quantization (still CPU, no training); then quantization-aware training (QAT) or quantization-aware distillation (QAD), which adapt the weights to the low-precision grid but reintroduce GPU training. These are deliberate future work, not part of the current stock-convert path.
</details>

## What you should check after the convert job

The convert job is automated, but verify the output before serving. The conversion logs (uploaded to S3 as `convert-logs.txt`) show the f16 conversion and the quantize step, including the final `model-<quant>.gguf` size, which should match the table above for your model and quant.

The real check is the next stage: the [report](report.md) runs the eval dataset through the served model end to end. That scorecard is where you see whether the model and quant you chose actually answer your task well once grounded by RAG. If accuracy is low, the usual levers are a larger base model, a higher quant, or better retrieval (the `embedding` task), in that order of effort.

## References

- [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). Why memory bandwidth determines inference speed, and why quantization matters for CPU deployment.
- [GGUF format specification](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md). The model format used by llama.cpp for CPU inference.
- [llama.cpp](https://github.com/ggerganov/llama.cpp). The engine Slemify uses to convert, quantize, and serve GGUF models on CPU.
- [Phi-2: The Surprising Power of Small Language Models](https://arxiv.org/abs/2309.05463) (Microsoft, 2023). Evidence that smaller models match larger ones on structured tasks.
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (NVIDIA, 2025). Research case for using SLMs under 10B parameters in production agentic systems.
- [Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning](https://arxiv.org/abs/2505.11574). Why low-bit quantization hurts reasoning far more than perplexity, the effect behind the auditor's q8_0 choice.
