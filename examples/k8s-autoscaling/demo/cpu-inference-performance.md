# Measuring quality and throughput of a CPU-only agentic pipeline on Amazon EKS

This report covers two questions about the Kubernetes autoscaling assistant that runs its models on CPU with Amazon EKS Auto Mode. First, does the assistant answer correctly across the questions it is meant to handle? Second, how many requests can it serve, and how does that number behave as load rises? We ran both tests against a live deployment and report what we measured.

The assistant answers Kubernetes autoscaling questions (Karpenter, KEDA, HPA, Spot) by routing each query through five model-backed steps. Four of those steps run small models on CPU: a triage classifier that routes the query, an embedding model that turns the query into a vector for retrieval, a cross-encoder reranker that scores retrieved documentation, and a 30B-parameter analyst that writes the answer. The fifth step, a faithfulness gate, calls a frontier model (Claude Sonnet 5 on Amazon Bedrock) to check that the analyst's answer is supported by the retrieved evidence. Only the gate leaves the CPU; everything else is a small model on a Graviton node.

## Test environment

- Cluster: Amazon EKS Auto Mode, eu-west-1.
- Analyst node: `m9gd.8xlarge` (Graviton, arm64). The analyst pod requests 16 vCPU and 40 GiB, pinned.
- Analyst server: llama.cpp serving a Q4_K_M GGUF of the 30B mixture-of-experts model, 16 threads, an 8192-token context, prompt caching on.
- Pipeline configuration under test: triage on the CPU classifier, embedding on the CPU retriever, reranking on, analyst on the CPU small model, gate on the frontier model.

We measured each model directly, sending requests to its service inside the cluster rather than through the orchestrator, so each number reflects that one model's capacity without pipeline noise. We generated load from an in-cluster job to avoid the measurement error a laptop port-forward would add.

## Part 1: Answer quality

We ran the eight questions in the assistant's evaluation set through the live `/query` endpoint once each and graded every answer against the authoritative points recorded for that question. The set is representative rather than adversarial: fair, documentation-answerable questions across the main topics, plus one valid-config confirmation, one out-of-scope question the assistant should decline, and one off-topic message it should reject.

All eight answers were correct.

| Question | Result | Latency | Frontier calls | Cost |
|---|---|--:|--:|--:|
| Karpenter consolidationPolicy values | correct | 12.0s | 1 | $0.0125 |
| Spot interruption handling / NTH | correct | 157.6s | 2 | $0.0365 |
| minValues behavior | correct | 58.0s | 1 | $0.0130 |
| Is this NodePool valid? | correct | 78.5s | 1 | $0.0162 |
| HPA CPU metricType values | correct | 53.2s | 1 | $0.0125 |
| KEDA scale-to-zero | correct (escalated) | 140.8s | 5 | $0.0991 |
| Max NodePools for a 5000-node cluster | correct (declined) | 64.4s | 1 | $0.0150 |
| Weather in Seattle (off-topic) | correct (rejected) | 0.0s | 0 | $0.0000 |

Three results are worth calling out. The consolidationPolicy answer named all three valid values (`WhenEmpty`, `WhenEmptyOrUnderutilized`, `Balanced`) with correct descriptions, which is the case a grounded answer has to get right. The out-of-scope question drew an honest decline rather than an invented number, because the documentation does not state an official maximum. The off-topic message was rejected at the CPU triage classifier in zero time for zero frontier cost, which is the cheapest path through the system.

The KEDA question escalated. The analyst's first draft did not satisfy the gate, so the assistant gathered live evidence, redrafted, escalated to the frontier model, and finally produced a calibrated answer that states only what the evidence supports and names what it could not confirm. The final answer was correct. This is the intended behavior when a small model cannot ground an answer: escalate, and if the escalated answer still cannot be supported, answer with stated uncertainty rather than confidence.

## Part 2: Throughput and concurrency

A single question about the assistant kept coming up: if one request takes over two minutes, does that mean the system serves only one request every two minutes? The short answer is no, and the reason is that latency and throughput are different measurements. Latency is how long one request takes. Throughput is how many requests complete per minute. A slow request does not hold the whole machine for its entire duration, so the two numbers do not track each other.

To measure throughput we sent a steady, closed-loop load at each model: a fixed number of worker threads, each sending requests back to back for a set window. We swept the number of workers (the concurrency) and recorded the completion rate and the latency at each level.

### The analyst is the bottleneck

The analyst sets the pipeline's throughput. We swept it with short 200-token answers.

| Concurrent requests | Latency (p50) | Throughput | Analyst decode |
|--:|--:|--:|--:|
| 1 | 3.4s | 17.4 req/min | 57.9 tok/s |
| 2 | 5.3s | 22.7 req/min | 75.8 tok/s |
| 4 | 8.6s | 27.1 req/min | 90.0 tok/s |
| 8 | 16.9s | 28.1 req/min | 93.6 tok/s |

A single stream serves about 17 requests per minute, not one. Adding concurrency raises throughput up to a ceiling near 28 requests per minute (about 94 tokens per second), and past roughly four concurrent requests the extra load only lengthens latency. It does not produce more tokens.

That ceiling is the node's memory bandwidth. On CPU, decode speed is memory bandwidth divided by the bytes read per token, and the total tokens per second across every in-flight request is capped by that bandwidth. Below the cap, running requests concurrently helps, because the analyst can process one request's prompt while decoding another's answer. At the cap, the requests share a fixed budget, so each one slows down as you add more.

The knee of the curve, where throughput is near its ceiling but latency has not yet climbed steeply, sits around two to four concurrent requests: roughly 27 requests per minute at a 9-second median latency for short answers.

### The CPU encoders and reranker are not the bottleneck

The other CPU models have far more headroom.

| Model | Role | Single stream | Saturated ceiling |
|---|---|--:|--:|
| Analyst (30B GGUF, llama.cpp) | writes the answer | 17 req/min | 28 req/min |
| Reranker (cross-encoder, 5 documents per call) | scores retrieved chunks | 407 req/min | ~400 req/min |
| Retriever (embedding encoder) | embeds the query | 4,130 req/min | 17,200 req/min |
| Triage (classifier encoder) | routes the query | 8,500 req/min | 20,300 req/min |

The reranker has about 14 times the analyst's throughput, and the encoders have 600 to 700 times. Sustained pipeline throughput is therefore the analyst's throughput, and only queries that reach the analyst cost meaningful time. A query the triage classifier rejects costs nothing on the frontier model and runs at tens of thousands per minute.

## How to state the throughput

State throughput as the bottleneck's rate at a stated latency, not as a single headline number and not as the inverse of a worst-case latency. For this deployment and short answers:

- One analyst pod serves about 17 requests per minute single-stream and saturates near 28 requests per minute.
- The estimate for a given answer length is: analyst throughput ≈ decode tokens per second (about 94 at saturation) divided by tokens generated per answer.
- Real answers are longer than the 200-token test answers, and some escalate or gather evidence, so per-pod throughput in production is lower. Plan for single-digit to low-double-digit answers per minute per analyst pod, and re-measure with your expected answer length.

The two-minute figure some early tests showed was a worst-case latency, from a large prompt, extended reasoning, and a live-evidence loop on a single request. It is not the service rate.

## Scaling levers

- To raise throughput per pod, give llama.cpp more decode slots with continuous batching (the `--parallel` setting). Several sequences then decode together and use the memory-bandwidth budget more fully, which raises aggregate tokens per second at the cost of higher per-request latency. The analyst under test used a single slot and still gained throughput up to four concurrent requests because prompt processing and decoding overlap.
- To raise throughput overall, add analyst replicas. EKS Auto Mode provisions the nodes, and each replica adds another unit of the rate above. Throughput scales with replicas while per-request latency stays the same.
- To lower latency, shorten answers, keep retrieved context tight so the cold prompt is smaller, and keep the model resident in memory (the deployment already pins it). An accelerator is the right lever when a single answer must be fast; CPU is the right lever when cost per query matters more than tail latency.

## Method and caveats

We generated load inside the cluster over service DNS, closed-loop, with one warmup request per target, 45 seconds per concurrency level for the analyst and 25 seconds for the faster models. Analyst decode rates come from each response's reported completion-token count; latencies are end to end at the service. These are point-in-time numbers on one node type with short prompts. Treat them as the shape of the curves and the order of magnitude, and re-run the sweep against your expected prompt and answer sizes for planning-grade figures.
