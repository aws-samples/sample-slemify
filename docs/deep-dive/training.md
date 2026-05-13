# Training Stage

The training stage is fully automated. Once you've verified your synthetic data (see [Data Stage](data.md)), Slemify handles model loading, fine-tuning, checkpoint management, quantization, and artifact upload. You don't need to write training scripts, pick hyperparameters, or manage GPU instances.

Your only decision at this stage is which base model to use and whether to enable Spot instances. Everything else is auto-sized based on the model and dataset.

## What happens during training

```
train.jsonl + eval.jsonl (from S3)
        │
        ▼
Download to GPU node
        │
        ▼
Load base model in 4-bit (QLoRA)
        │
        ▼
Fine-tune with Unsloth + SFTTrainer
        │
        ▼
Checkpoint sync to S3 (for Spot recovery)
        │
        ▼
Export to GGUF (quantized)
        │
        ▼
Upload adapter + GGUF to S3
```

1. **Download.** An init container pulls `train.jsonl` and `eval.jsonl` from S3 to the GPU node.
2. **Load.** The base model is loaded in 4-bit precision using Unsloth's optimized loader. This keeps the full model in GPU memory even on a single 16GB card.
3. **Fine-tune.** QLoRA adapters are trained using the SFTTrainer from Hugging Face's TRL library. Unsloth's custom Triton kernels accelerate the forward and backward passes.
4. **Checkpoint.** After every N steps (auto-sized based on model size), the adapter weights are synced to S3. If a Spot instance is reclaimed, the next pod resumes from the last checkpoint.
5. **Export.** The trained adapter is merged with the base model and exported to GGUF format with the configured quantization level.
6. **Upload.** The final GGUF file and training logs are uploaded to S3 for the serving stage.

## Choosing a base model

The base model is the starting point for fine-tuning. Slemify supports any HuggingFace model that Unsloth can load. The choice depends on your task complexity.

<details>
<summary>What is fine-tuning?</summary>

A base model (like a 3B parameter instruct model) has been pre-trained on trillions of tokens of general text. It understands language, grammar, and a broad range of topics. Fine-tuning takes that general knowledge and specializes it for your specific task by training on your domain-specific data.

Think of it like hiring a smart generalist and giving them a week of on-the-job training. They already know how to read and reason. You're teaching them your specific classification categories and output format.
</details>

For classification and routing tasks (the primary use case for Slemify), smaller models are usually better:

| Model size | Good for | Training time (500 pairs) | Inference speed (CPU) |
|-----------|----------|--------------------------|----------------------|
| 1-3B | Classification, routing, simple extraction | ~10 min on single GPU | Fast. Sub-second on most CPUs. |
| 7-8B | Multi-step extraction, tool calling, structured generation | ~30 min on single GPU | Moderate. 1-2s on CPU. |
| 13B+ | Complex reasoning, long-form generation | 1+ hours | Slow on CPU. Consider GPU serving. |

**The practical guidance:** start with the smallest model that could plausibly handle your task. For classification with a fixed set of labels, a 3B model is almost always sufficient. You can always scale up if accuracy is too low, but you can't get back the latency and cost savings of a smaller model.

<details>
<summary>Why not always use the biggest model?</summary>

Bigger models are slower and more expensive to serve. On CPU (where Slemify deploys for inference), latency is roughly proportional to parameter count because inference is [memory-bandwidth bound](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). A 3B model moves about 1.8GB of weights through memory per token. An 8B model moves about 4.5GB. The CPU spends most of its time waiting for data, not doing math.

For classification tasks, the extra parameters in a larger model don't help. The model needs to learn a mapping from input text to a small set of labels. A 3B model has more than enough capacity for that. Research from [Microsoft](https://arxiv.org/abs/2309.05463) and [NVIDIA](https://arxiv.org/abs/2506.02153) confirms that models under 10B parameters match or beat larger models on structured, repetitive tasks when fine-tuned for the specific domain.
</details>

## QLoRA: how fine-tuning works

Slemify uses [QLoRA](https://arxiv.org/abs/2305.14314) (Quantized Low-Rank Adaptation) for fine-tuning. This is worth understanding because it explains why training is fast, cheap, and safe.

<details>
<summary>What is QLoRA?</summary>

Traditional fine-tuning updates every parameter in the model. For a 3B model, that's 3 billion numbers to adjust, requiring massive GPU memory and risking "catastrophic forgetting" (the model loses its general language ability while learning your task).

QLoRA takes a different approach:

1. **Quantize** the base model to 4-bit precision. This shrinks it to fit in a single GPU's memory.
2. **Freeze** all the original weights. They don't change during training.
3. **Add small adapter layers** (LoRA) on top of the frozen model. These are tiny matrices (rank 32 by default) attached to the attention and feed-forward layers.
4. **Train only the adapters.** This is a fraction of the total parameters.

The result: you get a specialized model without modifying the base weights. The adapter learns your task-specific patterns while the base model retains its general language understanding.
</details>

The key QLoRA parameters Slemify configures:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `lora_r` | 32 | Rank of the adapter matrices. Higher = more capacity, more memory. 32 is a good balance for classification. |
| `lora_alpha` | 16 | Scaling factor for the adapter. Controls how much the adapter influences the output. |
| `target_modules` | All attention + feed-forward layers | Which layers get adapters. Slemify targets all of them for maximum task adaptation. |
| `load_in_4bit` | true | Base model precision. 4-bit keeps the full model in 16GB GPU memory. |

You don't need to tune these. They're set to values that work well across classification, extraction, and routing tasks. If you're curious about the research behind these defaults, the [QLoRA paper](https://arxiv.org/abs/2305.14314) covers the tradeoffs in detail.

## Auto-sized hyperparameters

Slemify's auto-sizer picks training hyperparameters based on your model size and dataset. You can override epochs in your config, but the defaults are designed to work without tuning.

| Parameter | How it's set | Rationale |
|-----------|-------------|-----------|
| Epochs | 5 for datasets under 10K samples, 3 for larger | Smaller datasets need more passes to learn the patterns. Larger datasets converge faster. |
| Learning rate | 2e-4 for models ≤7B, 1e-4 for larger | Smaller models tolerate higher learning rates. Larger models need gentler updates to avoid instability. |
| Warmup ratio | 0.1 (10% of training steps) | Gradually increases the learning rate at the start to avoid early instability. |
| Scheduler | Cosine | Smoothly decays the learning rate after warmup. Standard choice for fine-tuning. |
| Batch size | 2 (with 4x gradient accumulation) | Effective batch size of 8. Fits in 16GB GPU memory while providing stable gradient estimates. |
| Early stopping patience | 2 epochs | Stops training if the loss hasn't improved for 2 consecutive epochs. Prevents overfitting. |

**Overriding epochs:** If you want more or fewer training passes, set `training.epochs` in your config:

```yaml
training:
  epochs: 3    # Override the auto-sized default
  spot: true
```

When you set epochs to 6 or higher, the auto-sizer increases early stopping patience to 4 to give the model more room to converge.

## Spot GPU training

Training runs on Spot GPU instances by default (`training.spot: true`). This cuts GPU costs by 60-90% compared to on-demand pricing. The tradeoff is that Spot instances can be reclaimed at any time.

Slemify handles this with checkpoint-based recovery:

1. **Checkpoint sync.** After every N steps (500 for 3B models, 100 for 8B), the adapter weights are uploaded to S3 via a training callback.
2. **Spot interruption.** If the instance is reclaimed, the K8s Job's `backoffLimit: 6` ensures a new pod is scheduled on a fresh Spot instance.
3. **Automatic resume.** The new pod checks S3 for existing checkpoints. If found, it downloads the latest one and resumes training from that step.

The maximum work lost during a Spot interruption is the steps since the last checkpoint. For a 3B model with 500-step checkpoints, that's roughly 10-15 minutes of training.

**Karpenter integration.** The training Job uses node affinity to request GPU instances from the `g` and `p` families (NVIDIA T4, A10G, L4, A100). Karpenter provisions the cheapest available Spot instance that meets the GPU memory requirement (≥16GB). The `karpenter.sh/do-not-disrupt: true` annotation prevents Karpenter from voluntarily consolidating the node during training.

<details>
<summary>What if Spot capacity is unavailable?</summary>

If no Spot GPU instances are available in your region, the pod stays in Pending state until capacity appears. For time-sensitive workloads, you can set `training.spot: false` to use on-demand instances instead. The training code is identical; only the Karpenter NodePool capacity type changes.

In practice, Spot GPU availability varies by instance type and region. The g5 family (A10G) tends to have better Spot availability than p4 (A100) because demand is lower.
</details>

## Quantization and GGUF export

After training, Slemify exports the model to [GGUF format](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md) with quantization. This is what makes CPU inference possible.

<details>
<summary>What is quantization?</summary>

A model's weights are normally stored as 16-bit floating point numbers (2 bytes each). Quantization reduces the precision to fewer bits per weight, shrinking the model and making it faster to load from memory.

For a 3B parameter model:
- **F16 (no quantization):** ~6GB. Full precision, highest quality, slowest on CPU.
- **Q8_0 (8-bit):** ~3GB. Minimal quality loss, good balance.
- **Q4_K_M (4-bit):** ~1.8GB. Small quality loss, fastest on CPU.

The quality loss from Q4_K_M is negligible for classification tasks. The model is outputting a label from a fixed set, not generating creative prose. The precision of the weights matters much less when the output space is small.

Why this matters for CPU inference: the bottleneck is [memory bandwidth](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/), not compute. Every token generated requires reading the entire model's weights from RAM. A 1.8GB model reads 3x faster than a 6GB model, which translates directly to 3x faster token generation.
</details>

Slemify supports three quantization levels:

| Level | Config value | Model size (3B) | Quality | Speed |
|-------|-------------|-----------------|---------|-------|
| 4-bit | `q4_k_m` (default) | ~1.8GB | Excellent for classification | Fastest |
| 8-bit | `q8_0` | ~3GB | Near-lossless | Fast |
| No quantization | `f16` | ~6GB | Full precision | Slowest |

The default (`q4_k_m`) is the right choice for most classification and routing tasks. Use `q8_0` if you need higher precision for extraction tasks with numeric values. Use `f16` only if you're serving on GPU and don't need the size reduction.

Unsloth handles the full export pipeline in one step: merge the LoRA adapter with the base model, convert to GGUF format, and apply quantization. The resulting file is uploaded to S3 and used directly by the serving stage.

## Catastrophic forgetting

Fine-tuning can cause a model to "forget" its general language abilities while learning your specific task. This is called catastrophic forgetting.

QLoRA largely prevents this because the base model weights are frozen. Only the small adapter layers are trained. The base model's general knowledge stays intact.

Slemify's post-training evaluation includes an optional MMLU (Massive Multitask Language Understanding) check that compares the fine-tuned model against the base model on general knowledge questions. If the fine-tuned model scores significantly lower, it's a signal that the training data or hyperparameters need adjustment.

In practice, catastrophic forgetting is rare with QLoRA on classification tasks. The adapter learns to map inputs to labels without disrupting the base model's language understanding.

## Incremental retraining

When your domain evolves (new categories, updated data), you don't need to train from scratch. Slemify supports incremental retraining:

```yaml
training:
  incremental: true
  spot: true
```

Incremental mode makes three adjustments:

1. **Halves the learning rate.** The model is already close to a good solution. Smaller updates prevent overshooting.
2. **Reduces warmup.** From 10% to 5% of steps. Less warmup needed when starting from a trained state.
3. **Defaults to 2 epochs.** Just enough to incorporate new data without overfitting.

The training job checks S3 for an existing adapter from a previous run. If found, it resumes from that checkpoint. If not, it trains from scratch with the incremental hyperparameters.

## What you should check after training

Training is automated, but you should verify the output before moving to serving. The training logs (uploaded to S3 as `training-pod.log`) contain:

- **Loss curve.** The loss should drop steeply in the first epoch and then plateau. If it keeps dropping through the final epoch, you might benefit from more epochs. If it spikes or oscillates, the learning rate may be too high.
- **Final loss.** For classification tasks, a final loss under 0.5 is typical. Under 0.3 is good. If the loss is above 1.0, something is wrong with the data (check for format inconsistencies in your training JSONL).
- **Training time.** A 3B model on 500-800 pairs should train in under 15 minutes on a single GPU. If it's taking much longer, check that the GPU is actually being used (the `wait-for-gpu` init container logs will show the GPU type).

The report stage (next in the pipeline) runs a full accuracy evaluation against the eval dataset. That's where you'll see whether the model actually learned your classification task.

## References

- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023). The fine-tuning technique Slemify uses. Enables training on a single GPU by quantizing the base model to 4-bit.
- [Unsloth](https://github.com/unslothai/unsloth). Custom Triton kernels for 2-5x faster QLoRA training. Slemify uses Unsloth as the training backend.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021). The original LoRA paper. QLoRA builds on this by adding quantization.
- [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). Why memory bandwidth determines inference speed, and why quantization matters for CPU deployment.
- [GGUF format specification](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md). The model format used by llama.cpp for CPU inference.
- [Phi-2: The Surprising Power of Small Language Models](https://arxiv.org/abs/2309.05463) (Microsoft, 2023). Evidence that smaller models match larger ones on structured tasks.
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (NVIDIA, 2025). Research case for using SLMs under 10B parameters in production agentic systems.
