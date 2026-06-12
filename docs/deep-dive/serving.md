# Serving Stage

Slemify deploys your model as part of the pipeline so the report stage can evaluate it against a live endpoint. This deployment is real and production-quality, but its primary purpose within Slemify is validation.

It also serves as a reference for how to deploy SLMs in your own environment. The patterns here (llama.cpp on CPU, Karpenter for node provisioning, SOCI for fast image pulls, KEDA for autoscaling) are designed to be studied, copied, or adapted. If you already have an inference platform, take the GGUF model from S3 and plug it in. If you don't, this is a solid starting point.

The GGUF model that Slemify produces can be served with any compatible runtime: llama.cpp, vLLM, Ollama, or anything that reads GGUF files.

## Two serving paths

How a model is served depends on its `project.task`:

- **Generation** (`task: generation`) → **llama.cpp + GGUF**. The rest of this
  document describes this path: S3-mounted GGUF, llama.cpp flags, KEDA scaling on
  queue depth, TTFT/streaming. This is the bulk of the serving surface.
- **Classification** (`task: classification`) → **encoder + head, served via ONNX**.
  A lean CPU pod (onnxruntime + tokenizers, **no torch**) downloads the project's
  `encoder.onnx`, `tokenizer.json`, and `head.json` from S3, embeds the query
  with ONNX Runtime (CLS pooling + L2 normalize), applies the logistic head, and
  returns a label. The encoder is exported to ONNX by the training job, so the
  embeddings match training exactly and serving carries no heavyweight ML deps.

The classifier serving pod deliberately exposes the **same OpenAI-compatible
`/v1/chat/completions` contract** as the generative path, returning
`"<label>|<confidence>"` as the message content (confidence is the softmax
probability mapped to high/medium/low). This makes a classifier a drop-in
replacement for a generative router: an orchestrator pointed at the inference
Service needs no changes when you swap one for the other. The response also
includes the raw probability under a `slemify` field for clients that want the
numeric score.

Because the classifier embeds in-process and applies a tiny matrix, its latency
is dominated by the encoder forward pass (~25ms for bge-base on CPU) — there is
no token-by-token decode phase, so the latency-planning tables below (which model
the generation decode loop) do not apply to it.

The remainder of this document covers the generation (llama.cpp + GGUF) path.

## What happens in the reference deployment

```
GGUF model file (in S3)
        │
        ▼
S3 mount (preferred) or init container download
        │
        ▼
llama.cpp server starts with auto-sized config
        │
        ▼
Readiness probe confirms the server is healthy
        │
        ▼
Prometheus metrics available at /metrics
        │
        ▼
Report Job evaluates accuracy against eval data
```

1. **Model loading.** Slemify auto-detects whether the [Mountpoint for Amazon S3 CSI driver](https://docs.aws.amazon.com/eks/latest/userguide/s3-csi-create.html) is installed. If it is, the GGUF file is mounted directly from S3 as a read-only PersistentVolume. llama.cpp reads it via mmap with no download step. If the CSI driver is not installed, an init container downloads the GGUF file from S3 into an emptyDir volume (the fallback behavior).
2. **Server start.** The llama.cpp server loads the GGUF model and exposes an OpenAI-compatible API on port 8080. A Prometheus-compatible metrics endpoint is enabled at `/metrics` for observability and autoscaling.
3. **Health checks.** Readiness and liveness probes hit the `/health` endpoint. The Deployment won't receive traffic until the model is loaded and responding.
4. **Report.** A K8s Job runs the production readiness report against the live endpoint (covered in the [Report Stage](report.md)).

### S3 mount vs init container download

| Aspect | S3 Mount (Mountpoint CSI) | Init Container Download |
|--------|--------------------------|------------------------|
| Pod startup | Near-instant (no download) | Depends on model size (79s for 8.7GB) |
| First inference | Slower (cold pages fetched from S3) | Normal (model already in memory) |
| Subsequent requests | Fast (pages cached in kernel) | Fast (model in memory) |
| Pod restart | No re-download needed | Full re-download |
| Local disk usage | None (mmap from S3) | Model size in emptyDir |
| Prerequisite | Mountpoint CSI driver installed | None |

The S3 mount approach is preferred for production because it eliminates the download bottleneck during scaling events. When KEDA scales up a new replica, the pod starts immediately and begins serving (with slightly higher latency on the first few requests as pages are faulted in from S3). The init container approach is simpler and works without any additional cluster setup.

With S3 mount, the `--mlock` flag is important. Without it, the kernel can evict model pages from the page cache under memory pressure (other pods on the same node, KV cache growth). When pages are evicted, subsequent requests must re-fetch them from S3 over the network, degrading throughput from 55 tok/s to as low as 4 tok/s. The `--mlock` flag locks the model in RAM at startup, preventing eviction entirely.

## Why CPU inference

GPU inference is faster per request, but for the workloads Slemify targets (classification, routing, extraction), CPU inference is the better economic choice.

**The key rule: output token count determines if CPU is viable, not input context size.** Prefill (processing input) is compute-bound and fast even for long inputs (200-300ms for 2000 tokens). Decode (generating output) is memory-bandwidth bound and slow (each token requires reading the full model weights from RAM). For classification tasks that output 2-10 tokens, total latency stays within SLA bounds. For free-form generation of 500+ tokens, total latency grows to 30-70 seconds on an 8B model.

The reasoning comes down to utilization. A GPU instance costs 3-10x more per hour than a comparable CPU instance. To justify that cost, the GPU needs to stay busy. Classification requests are short (small input, tiny output), so each request uses the GPU for milliseconds and then it sits idle. Unless you're batching hundreds of concurrent requests, the GPU is underutilized and you're paying for idle capacity.

CPU instances, especially Spot, cost a fraction of GPU instances and scale horizontally. Adding a replica is cheap. Removing one when traffic drops is instant. There's no GPU scheduling contention, no NVIDIA driver compatibility issues, and near-unlimited Spot capacity for CPU instance families.

<details>
<summary>The memory bandwidth explanation</summary>

Inference speed on any hardware is determined by how fast the model's weights can move from memory to the compute units. This is called memory bandwidth, and it's the bottleneck for transformer models during the decode phase (generating tokens one at a time).

A quantized 3B model (Q4_K_M) is about 1.8GB. Every token generated requires reading those 1.8GB from RAM. On a modern CPU with ~50 GB/s memory bandwidth, that's roughly 36ms per token just for the memory transfer. The actual math (matrix multiplications) takes a fraction of that time. The CPU spends most of its time waiting for data.

This is why quantization matters so much for CPU inference. A Q4_K_M model (1.8GB) reads 3x faster than an F16 model (6GB) from the same memory bus. The speedup is nearly linear because the bottleneck is data movement, not computation.

For a deeper treatment of this topic, see [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/).
</details>

### CPU architectures for inference

Since inference is memory-bandwidth bound, the CPU architecture that moves data fastest wins. All three major server CPU families work well with llama.cpp, but they have different strengths:

## Latency planning: how input size affects response time

Use this table to estimate total response time based on your input token count. These numbers are based on tests we have run with llama.cpp and Q4_K_M quantization on a standard compute-optimised CPU instance (16 vCPUs, 8 cores allocated per model).

### 4B model (classification/routing tasks, 8 CPU cores)

Prompt throughput: ~367 tokens/sec. Generation: ~28 ms/token.

| Input | Tokens (approx) | Prompt eval | + 32 output tokens | **Total** |
|-------|-----------------|-------------|-------------------|-----------|
| Short query (1-2 lines) | 50 | 0.1s | 0.9s | **1.0s** |
| Config snippet (20 lines YAML) | 200 | 0.5s | 0.9s | **1.5s** |
| Email body (~500 words) | 700 | 1.9s | 0.9s | **2.8s** |
| Email + MCP context (~2000 words) | 2,800 | 7.6s | 0.9s | **8.5s** |
| Full thread + tool results (~5000 words) | 7,000 | 19.1s | 0.9s | **20s** |

### 8B model (structured generation tasks, 8 CPU cores)

Prompt throughput: ~220 tokens/sec. Generation: ~51 ms/token.

| Input | Tokens (approx) | Prompt eval | + 300 output tokens | **Total** |
|-------|-----------------|-------------|---------------------|-----------|
| Short query + RAG (2 docs) | 500 | 2.3s | 15.3s | **17.6s** |
| Medium query + RAG (3 docs) | 800 | 3.6s | 15.3s | **18.9s** |
| Long query + RAG (5 docs) | 1,500 | 6.8s | 15.3s | **22.1s** |

### Formula

```
Total time = (input_tokens / prompt_throughput) + (output_tokens x ms_per_token / 1000)
```

**Practical guidance:**
- For **routing/classification** (short output): keep input under ~1000 tokens for sub-3.5s response. If your input is larger, truncate to the most relevant section before sending to the SLM.
- For **structured generation** (long output): the generation phase dominates. Input size matters less because even doubling the input only adds 2-3s, while the 15s generation time is fixed.
- For **batch inference**: latency doesn't matter. Process overnight. CPU is viable for any input size when you're optimizing for cost, not speed.

| CPU family | Memory bandwidth | Key advantage for inference |
|-----------|-----------------|---------------------------|
| [AWS Graviton4](https://aws.amazon.com/ec2/graviton/) (Arm Neoverse V2) | 12x DDR5-5600 channels | Lowest cost per core-hour on AWS. 75% more bandwidth than Graviton3. |
| [AMD EPYC Turin](https://www.amd.com/en/products/processors/server/epyc/9005-series.html) (Zen 5) | 12x DDR5-6000 channels, up to 614 GB/s | Highest channel count and bandwidth per socket. Strong Spot availability on AWS (m7a, c7a families). |
| [Intel Xeon 6 Granite Rapids](https://www.intel.com/content/www/us/en/products/platforms/details/granite-rapids.html) | 8x DDR5-6400 channels, MRDIMM option at 8800 MT/s | MRDIMM support can push bandwidth beyond standard DDR5 limits. AMX tile registers accelerate matrix operations. |

Slemify's Karpenter NodePool allows both arm64 and amd64 architectures and uses on-demand capacity. Karpenter evaluates all eligible instance types across families and picks the cheapest option that meets the CPU and memory requirements.

### Preferring latest generation instances with NodeOverlays

Newer instance generations (e.g., Graviton4 c8g vs Graviton3 c7g) offer better memory bandwidth and price-performance for inference. Slemify uses [Karpenter NodeOverlays](https://karpenter.sh/docs/concepts/nodeoverlays/) (alpha) to prefer the latest generation by penalizing older generations through price adjustments:

| Generation | Penalty | Effect |
|-----------|---------|--------|
| Gen 5 (c5, m5, r5) | +45% | Strongly deprioritized |
| Gen 6 (c6g, m6i, r6g) | +30% | Deprioritized |
| Gen 7 (c7g, m7i, r7g) | +15% | Slightly deprioritized |
| Gen 8 (c8g, m8g, r8g) | No penalty | Preferred |

With on-demand capacity, this gives deterministic selection. Karpenter always picks the lowest perceived price, which is the latest generation. Both arm64 (Graviton) and amd64 are eligible, but Graviton instances are typically cheaper per core, so they're naturally preferred.

If cost is the primary concern, you can switch the NodePool to Spot capacity. NodeOverlays still apply, but EC2 Fleet uses `capacity-optimized-prioritized` for Spot, where capacity availability can override your generation preferences. You'll still get a preference for latest gen, but not a guarantee. EC2 may select an older generation if it has better Spot capacity.

NodeOverlays require the `NodeOverlay` feature gate to be enabled in Karpenter (`settings.featureGates.nodeOverlay=true`). If the feature gate is not enabled, Slemify skips the overlays gracefully and Karpenter selects instances based on pure price optimization.

For a detailed comparison of how these architectures handle the instruction-data-shape triangle for inference workloads, see [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/).

## llama.cpp configuration

Slemify uses [llama.cpp](https://github.com/ggerganov/llama.cpp) as the inference engine. It loads GGUF models and exposes an OpenAI-compatible HTTP API (`/v1/chat/completions`).

The server is configured with flags tuned for classification workloads:

| Flag | Value | Why |
|------|-------|-----|
| `--ctx-size 512` | Small context window | Classification inputs are short. A smaller context window uses less memory and reduces per-request overhead. For free-form models, this is set dynamically based on training data output length. |
| `--flash-attn on` | Flash Attention enabled | Reduces memory usage for the attention computation. Meaningful even on CPU for longer inputs. |
| `--batch-size 512` | Prompt processing batch | How many tokens are processed at once during the prompt phase. 512 is a good balance between throughput and memory. |
| `--repeat-penalty 1.1` | Mild repetition penalty | Prevents the model from getting stuck in loops when generating output. |
| `--min-p 0` | No minimum probability filter | Allows the model to consider all tokens. For classification, the output is constrained enough that aggressive filtering isn't needed. |
| `--mlock` | Lock model in RAM | Prevents the kernel from evicting model pages under memory pressure. Without this, idle pods can degrade from 55 tok/s to 4 tok/s as pages are evicted and must be re-fetched from S3. |
| `--cache-prompt` | Reuse KV cache across requests | Keeps the KV cache from the previous request in memory. If the next request shares a prefix (e.g., same system instruction), the cached tokens are reused without re-processing. Reduces TTFT on subsequent requests when prompts share common prefixes. |

For models that support a "thinking" mode, Slemify adds `--reasoning-budget 0` to disable it. The fine-tuned model produces output directly without needing internal reasoning tokens. Thinking mode adds latency (5-10s of thinking overhead) without improving output quality for fine-tuned models.

<details>
<summary>Tuning context window size</summary>

The context window (`--ctx-size`) determines how many tokens the model can process in a single request. For classification, inputs are typically under 200 tokens and outputs are 2-3 tokens. A context window of 512 is more than enough.

Larger context windows use more memory (for the KV cache) and slightly increase per-request latency. If your inputs are consistently short, keeping the context window small is free performance.

If your use case involves longer inputs (full documents, long log entries), you may need to increase this. The tradeoff is linear: doubling the context window roughly doubles the KV cache memory usage.
</details>

## Auto-sized resources

The auto-sizer maps your model size and quantization level to CPU, memory, and thread count for the inference pod. Karpenter then picks the cheapest instance that satisfies those resource requests.

| Model size | CPU request | Memory request | Threads |
|-----------|------------|---------------|---------|
| ≤3B | 4 cores | 6Gi (Q4_K_M) / 12Gi (F16) | 4 |
| ≤8B | 8 cores | 16Gi (Q4_K_M) / 24Gi (F16) | 8 |
| >8B | 16 cores | 24Gi (Q4_K_M) / 40Gi (F16) | 16 |

The memory request accounts for the model file plus the KV cache and runtime overhead. The CPU request determines how many threads llama.cpp uses for matrix operations. More threads help up to the point where memory bandwidth saturates, after which adding threads provides no benefit.

**Thread count matters.** Setting threads higher than the number of physical cores (ignoring hyperthreads) can actually hurt performance due to cache contention. The auto-sizer sets threads equal to the CPU request, which maps to physical cores on most instance types.

**Models larger than 8B.** The auto-sizer supports models up to 30B+ parameters on CPU. Larger models work but with proportionally higher latency (more weights to read per token). For classification and routing tasks with short output, models up to 30B are viable on CPU if the response time fits your SLAs. For most classification tasks, 3-8B is the sweet spot: fast enough for real-time use, large enough for multi-class accuracy.

## Karpenter and instance selection

The serving stage creates a Karpenter NodePool that provisions CPU instances for inference. The NodePool is configured to:

- **Allow multiple instance families.** The `c` (compute-optimized), `m` (general-purpose), and `r` (memory-optimized) families are all eligible. Karpenter picks the cheapest available option.
- **Prefer Spot.** Both Spot and on-demand are allowed, with Karpenter preferring Spot for cost savings.
- **Allow arm64 and amd64.** llama.cpp runs on both architectures. AWS Graviton (arm64) instances are typically cheaper per core-hour, but AMD EPYC and Intel Xeon instances are also eligible. Karpenter picks the cheapest available option across all architectures.
- **Exclude tiny instances.** Nano, micro, and small sizes are excluded because they don't have enough memory or CPU for model serving.
- **Consolidate when idle.** The `WhenEmptyOrUnderutilized` consolidation policy removes nodes that aren't carrying useful workload, keeping costs down during low-traffic periods.

The NodePool uses a dedicated taint (`slemify.io/slm: NoSchedule`) so inference pods don't compete with other workloads for node resources.

## Autoscaling with KEDA

Slemify deploys the inference endpoint with a single replica. Horizontal scaling is your responsibility, and [KEDA](https://keda.sh) (Kubernetes Event-Driven Autoscaling) is the recommended approach.

llama.cpp exposes a Prometheus-compatible metrics endpoint at `/metrics` (enabled by default in Slemify). The metrics that matter for scaling are:

| Metric | What it measures | Why it matters |
|--------|-----------------|---------------|
| `llamacpp:requests_processing` | Requests currently being handled | Shows current load per replica |
| `llamacpp:requests_deferred` | Requests queued, waiting for a slot | Leading indicator of latency degradation |
| `llamacpp:kv_cache_usage_ratio` | KV cache fullness (0.0 to 1.0) | Approaching 1.0 means the server will start rejecting requests |
| `llamacpp:predicted_tokens_seconds` | Token generation throughput | Useful for monitoring, less useful for scaling |

### Why queue depth, not CPU utilization

CPU utilization is a saturation metric. It spikes *after* latency has already degraded. By the time CPU-based autoscaling reacts, your users are already waiting.

Queue depth (`requests_deferred`) is a leading indicator. It rises *before* latency degrades, because requests start queuing when all processing slots are busy. Scaling on queue depth means new replicas are provisioned while existing ones are still responding normally.

GPU utilization is even worse as a scaling signal. For inference workloads, GPU utilization sits at 100% regardless of whether the server is handling 1 request or 50. It provides no signal about capacity.

### The combined formula approach

A single metric with a flat threshold is fragile. A better approach is combining `requests_processing` (how busy are we?) with `requests_deferred` (are we falling behind?), weighting the queue metric heavily:

```yaml
# Example KEDA ScaledObject for llama.cpp inference
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: my-project-inference
spec:
  scaleTargetRef:
    name: my-project-inference
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
    - type: kedify-otel    # or prometheus
      name: running
      metadata:
        metricQuery: 'sum(llamacpp:requests_processing)'
        threshold: '5'
    - type: kedify-otel
      name: deferred
      metadata:
        metricQuery: 'sum(llamacpp:requests_deferred)'
        threshold: '1'
  advanced:
    scalingModifiers:
      formula: "running + (deferred * 10)"
      target: "25"
      activationTarget: "5"
```

The formula `running + (deferred * 10)` means even 3 queued requests push the combined metric to 55, well above the target of 25. Scaling kicks in aggressively before latency degrades. The `activationTarget` of 5 prevents noise from triggering unnecessary scale-from-zero events.

### OTel for scaling, Prometheus for monitoring

The standard KEDA Prometheus scaler works, but it has an inherent delay. Prometheus scrapes metrics every 15-30 seconds, then KEDA polls Prometheus on its own interval. By the time the metric reaches KEDA, you can be 30-60 seconds behind reality.

For faster scaling decisions, consider the [Kedify OTel Scaler](https://kedify.io/scalers/otel) (open-source). Instead of the pull model (Prometheus scrapes pods, KEDA queries Prometheus), an OTel Collector sidecar pushes metrics directly from each pod to the scaler via OTLP. The scaler stores them in a lightweight in-memory buffer and exposes them to KEDA over gRPC. The result is near real-time metrics in the scaling path.

You can keep Prometheus running for dashboards and alerting. OTel takes over only the scaling path where freshness matters most.

<details>
<summary>Setting up the OTel approach</summary>

The OTel scaling path has three components:

1. **OTel Collector sidecar.** Injected into each inference pod by the OTel Operator. It scrapes the local `/metrics` endpoint and pushes to the Kedify OTel Scaler via OTLP.
2. **Kedify OTel Scaler.** Receives metrics from all sidecars, stores them in memory, and exposes them to KEDA as an external scaler.
3. **KEDA ScaledObject.** Uses the `kedify-otel` trigger type instead of `prometheus`.

The OTel Operator handles sidecar injection automatically via a pod annotation. The sidecar only forwards the metrics you specify, so it's lightweight. All three components deploy from a single Helm chart.

For a complete working example with KEDA, Karpenter, and OTel-based scaling, see the [kedify-on-eks-blueprint](https://github.com/kedify/kedify-on-eks-blueprint) project.
</details>

## Time to first token and streaming

For applications that stream responses to users (chat interfaces, progressive rendering), the metric that matters is Time to First Token (TTFT), not total generation time. Once the first token arrives, the user sees activity and perceives the system as responsive. The remaining tokens stream in progressively while the user reads.

This changes how you think about latency budgets. A 14-second total generation time sounds slow, but if the first token arrives in 1.2 seconds and the rest streams token by token, the user experience is comparable to a 3-second LLM API call that delivers the full response at once. The streaming approach turns a latency problem into a perceived-performance advantage.

TTFT is determined by the prompt processing phase: how fast the model can ingest the input tokens and produce the first output token. For CPU inference, this depends on input length, context window size, and memory bandwidth. Short inputs (classification, routing) have sub-second TTFT. Longer inputs (full YAML configs with RAG context) take 1-2 seconds for prompt processing before the first token appears.

When designing applications on top of SLM inference, optimize for TTFT:
- Stream responses via SSE or WebSocket rather than waiting for full completion
- Show progress indicators during the prompt processing phase (triage result, RAG retrieval status)
- Keep the prompt as short as possible (trim RAG context to the most relevant chunks)
- Use a smaller, faster model for the routing step (3-4B) and a larger model (8B) only for generation

## Application startup and readiness

When KEDA scales up a new replica or Karpenter provisions a fresh node, the new pod must be fully warm before receiving traffic. Without warmup, the first request pays the cost of TLS handshake to external services (Bedrock for embeddings), connection pool initialization, and SLM prompt cache population. This can add 3-10 seconds to the first request, which is unacceptable when the steady-state TTFT is 1.2 seconds.

The pattern: run a warmup sequence during pod startup that exercises the full inference path, and gate the readiness probe on warmup completion.

```python
# Simplified warmup pattern
_ready = False

@app.on_event("startup")
async def warmup():
    global _ready
    # 1. Call each SLM with a short prompt (warms model cache + connection)
    # 2. Call Bedrock embedding (warms TLS + credential fetch via Pod Identity)
    # 3. Set ready flag
    _ready = True

@app.get("/health")
async def health():
    if not _ready:
        return JSONResponse({"status": "warming up"}, status_code=503)
    return {"status": "ok"}
```

The readiness probe points at `/health`, which returns 503 until warmup completes. Kubernetes won't route traffic to the pod until it passes. The liveness probe points at a different path (like `/`) that always returns 200, so the pod isn't killed during the warmup window.

### Measured startup timeline (fresh node, no image cache)

| Phase | Duration | Cumulative |
|-------|----------|-----------|
| Karpenter provisions node | ~24s | 24s |
| Image pull (SOCI, ~150MB) | ~9s | 33s |
| Warmup (SLMs + Bedrock) | ~22s | 55s |
| **Pod Ready, first query TTFT** | **1.2s** | **56s** |

After the pod is Ready, first and subsequent queries have identical TTFT (1.2s). There is no cold-start penalty for the user because the warmup absorbed it.

For steady-state scaling (node already exists, image cached), the pod is Ready in ~17 seconds. The warmup dominates because it waits for the SLM pods to generate 64 tokens each (confirming the model is loaded and the inference path is exercised end to end).

### Why 64 tokens, not 1

A 1-token warmup only loads the model weights into memory. It doesn't exercise the decode loop or warm the CPU caches for sequential token generation. A 64-token warmup forces the model through both the prompt processing phase and the generation phase, ensuring the first real query doesn't pay any hidden initialization cost.

## Pod Disruption Budget

A PodDisruptionBudget (PDB) with `minAvailable: 1` ensures that at least one inference replica stays running during voluntary disruptions (node upgrades, Karpenter consolidation, cluster maintenance). This prevents downtime during routine operations.

The PDB does not protect against involuntary disruptions like Spot reclamation. For high-availability deployments, run multiple replicas so that losing one Spot instance doesn't cause an outage.

## Faster container startup with SOCI

When a new node is provisioned (scaling up, Spot replacement), the container runtime needs to pull the container image before the pod can start. The default containerd behavior downloads and unpacks each image layer sequentially and fully before starting the container. For multi-GB images like the Unsloth training container, this can add 30-60 seconds to pod startup.

Slemify uses [Bottlerocket](https://bottlerocket.dev/) as the node OS, which has native support for the [SOCI snapshotter](https://github.com/awslabs/soci-snapshotter) (Seekable OCI). SOCI is enabled via simple TOML settings in the EC2NodeClass userData, with no shell scripts or manual installation. It replaces the default sequential pull with parallel chunk-based downloads:

- **20 concurrent downloads per image.** Instead of pulling one layer at a time, SOCI pulls chunks from multiple layers simultaneously.
- **16MB chunk size.** Each download is a 16MB piece of a layer, allowing fine-grained parallelism.
- **12 concurrent unpacks.** Decompression happens in parallel with downloads, overlapping I/O and CPU work.

This is configured automatically on both the CPU (inference) and GPU (training) NodePools. No changes to your container images are needed. SOCI works with standard OCI images from any registry, including Amazon ECR.

<details>
<summary>How SOCI works</summary>

The default containerd snapshotter (OverlayFS) downloads each image layer as a single blob, decompresses it fully to disk, then moves to the next layer. This is sequential and blocking: the container can't start until every layer is fully unpacked.

SOCI v0.11.0+ introduces a "parallel-pull-unpack" mode that breaks each layer into chunks and downloads them concurrently. It uses a temporary file buffer instead of an in-memory one, which allows the store and decompression operations to overlap. The result is faster image pulls, limited only by network bandwidth and disk I/O.

This is similar in concept to the [multipart layer fetch](https://github.com/containerd/containerd/pull/10177) introduced in containerd 2.1.0, but available today as a snapshotter plugin.

For more details on container startup optimization strategies, see the [AI on EKS guidance on accelerating pull processes](https://awslabs.github.io/ai-on-eks/docs/guidance/container-startup-time/accelerate-pull-process).
</details>

## Latency optimization checklist

If inference latency is higher than expected, check these in order:

1. **Quantization level.** Q4_K_M is the fastest. If you're using Q8_0 or F16, the model is larger and reads slower from memory.
2. **Context window.** If `--ctx-size` is set higher than needed, reduce it. Smaller context = less KV cache memory = faster per-request processing.
3. **Model size.** A 3B model is roughly 2-3x faster than an 8B model on the same hardware. If your task is classification, a 3B model is almost always sufficient.
4. **Input length.** Longer inputs take longer to process (the prompt phase scales linearly with token count). If possible, trim or preprocess inputs before sending them to the model.
5. **Thread count.** Check that threads match physical cores. Too many threads cause contention. Too few leave CPU capacity unused.
6. **Instance generation.** Newer CPU generations have higher memory bandwidth, which directly translates to faster inference. Graviton4 (12x DDR5-5600 channels) provides roughly 75% more bandwidth than Graviton3. AMD EPYC Turin (12x DDR5-6000 channels, up to 614 GB/s per socket) and Intel Xeon 6 Granite Rapids (8x DDR5-6400 channels, with optional MRDIMM at 8800 MT/s) are also strong choices. If you're on older instance types, newer generations will be meaningfully faster for the same or lower cost. See [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/) for a detailed comparison.

## The OpenAI-compatible API

The llama.cpp server exposes a standard OpenAI-compatible API. Any client that works with the OpenAI API works with Slemify's inference endpoint:

```bash
curl http://<service>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Classify this: my unit wont boot"}
    ]
  }'
```

This means agents, orchestrators, and applications can call the SLM endpoint without any Slemify-specific client code. It's a standard HTTP API that returns JSON.

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp). The inference engine Slemify uses for CPU deployment. Supports GGUF models with quantization.
- [Mountpoint for Amazon S3 CSI Driver](https://github.com/awslabs/mountpoint-s3-csi-driver). Mounts S3 buckets as read-only filesystems in Kubernetes pods. Slemify uses it to serve GGUF models directly from S3 via mmap, eliminating download time during pod startup and scaling.
- [Silicon, Memory, and Modern Inference](https://cmanaha.github.io/tech-deep-dives/silicon-memory-inference/). Why memory bandwidth (not FLOPs) determines inference speed on both CPU and GPU.
- [Karpenter](https://karpenter.sh). Just-in-time node provisioning for Kubernetes. Slemify uses it to provision the cheapest Spot instances for both training and inference.
- [KEDA](https://keda.sh). Kubernetes Event-Driven Autoscaling. Scales inference replicas based on concurrent request metrics from Prometheus.
- [SOCI Snapshotter](https://github.com/awslabs/soci-snapshotter). Parallel chunk-based container image pulls for faster pod startup. Configured automatically on all Slemify nodes.
- [AI on EKS: Accelerating Container Startup](https://awslabs.github.io/ai-on-eks/docs/guidance/container-startup-time/accelerate-pull-process). Guidance on SOCI, Nydus, and image preloading strategies for AI workloads on EKS.
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (NVIDIA, 2025). SLMs offer 10-30x lower inference costs per token, making them the sustainable choice for high-frequency agentic deployment.
- [GGUF format specification](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md). The model format optimized for CPU inference with llama.cpp.
