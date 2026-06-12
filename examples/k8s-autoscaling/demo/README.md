# K8s Autoscaling Expert Demo

A multi-agent system that audits Kubernetes autoscaling configurations using fine-tuned SLMs on CPUs, RAG for knowledge grounding, and LLM escalation for out-of-domain queries.

The demo shows that CPUs handle the full AI inference pipeline (routing, retrieval, generation) without GPUs, delivering domain-accurate results at a fraction of the cost of LLM APIs.

## Architecture

```
User submits a K8s config question via chat UI
        |
        v
[Triage SLM - 4B, CPU, ~1.5s]
  Classifies intent: karpenter_config, keda_config, hpa_config,
  pdb_disruption, spot_interruption, multi_resource, or noise
        |
        |-- noise --> "This doesn't look like a K8s autoscaling question."
        |
        |-- low confidence --> [LLM API + RAG] Bedrock Sonnet 4.5 with docs context
        |
        |-- high confidence -->
                |
                v
        [Embedding - bge-base-en-v1.5, CPU, ~5ms]
          Embeds the query locally (768d), no external API call
                |
                v
        [OpenSearch - Vector DB, CPU, ~100ms]
          Retrieves 10 candidate Karpenter/KEDA doc chunks
                |
                v
        [Reranker - bge-reranker-base cross-encoder, CPU]
          Scores all 10 candidates, keeps the best 2
                |
                v
        [Auditor SLM - 8B, CPU, streaming ~14s]
          Structured analysis with RAG context:
          Error type, severity, root cause, remediation
                |
                v
        Response streamed to user token by token
```

## Components

| Component | Runtime | Instance | Role |
|-----------|---------|----------|------|
| Chat UI | Static web app | Any | User interface with markdown rendering |
| Orchestrator | Python FastAPI | CPU pod | Routes between triage, RAG, auditor, LLM |
| Triage SLM | llama.cpp | c8g (Graviton4 CPU) | Intent classification, 1.5s |
| Embedding | sentence-transformers (bge-base-en-v1.5) | c8g (Graviton4 CPU) | In-cluster query/doc embeddings, 768d |
| Reranker | sentence-transformers (bge-reranker-base) | c8g (Graviton4 CPU) | Cross-encoder re-ranks top-10 candidates to best 2 |
| OpenSearch | OpenSearch k-NN | CPU pod | Vector search over 3900+ doc chunks |
| Auditor SLM | llama.cpp | c8g (Graviton4 CPU) | Structured config analysis, streamed |
| LLM API | Bedrock (Sonnet 4.5) | Managed | Fallback for low confidence queries |

## Knowledge Base

~3900 chunks indexed from:
- Karpenter v1 docs (API reference, concepts, troubleshooting)
- KEDA v2.19 docs (ScaledObject, triggers, authentication)
- AWS EKS Best Practices (autoscaling section)
- 17 AWS blog posts (Karpenter v1.0, Spot consolidation, Graviton migration, KEDA + Prometheus, etc.)

Embedding model: bge-base-en-v1.5 (768 dimensions), served in-cluster on CPU via sentence-transformers. No external API call for embeddings.

## Demo Prompts (Tested)

### 1. High confidence (Auditor SLM responds)

**NodePool limits:**
```
pods are stuck in Pending but karpenter isn't launching new nodes. no errors in the karpenter logs, it just says "can't create any more capacity." we checked and our NodePool has limits set:

apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: default
spec:
  template:
    spec:
      requirements:
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values: ["c", "m"]
        - key: karpenter.k8s.aws/instance-generation
          operator: Gt
          values: ["4"]
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: default
  limits:
    cpu: "50"
    memory: 200Gi
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 60s

kubectl get nodepool shows we're at 48 CPU used out of 50 limit. so karpenter won't provision more. but we have 20 pending pods. should we increase the limits?
```

**Deprecated API (ChatGPT gave wrong config):**
```
hey team, need some help with karpenter on our EKS 1.31 cluster. we're trying to get GPU nodes provisioned for our ML training workloads. i asked ChatGPT for a karpenter config and it gave me this:

apiVersion: karpenter.sh/v1alpha5
kind: Provisioner
metadata:
  name: gpu-provisioner
spec:
  requirements:
    - key: node.kubernetes.io/instance-type
      operator: In
      values: ["p3.2xlarge", "p3.8xlarge", "g4dn.xlarge", "g4dn.2xlarge"]
    - key: karpenter.sh/capacity-type
      operator: In
      values: ["on-demand"]
  limits:
    resources:
      cpu: 256
      memory: 1024Gi
      nvidia.com/gpu: 32
  provider:
    instanceProfile: KarpenterNodeInstanceProfile-ml-cluster
    subnetSelector:
      karpenter.sh/discovery: ml-cluster
    securityGroupSelector:
      karpenter.sh/discovery: ml-cluster
```

**minValues (compare with ChatGPT):**
```
my NodePool has this requirement but pods are still pending:

apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: default
spec:
  template:
    spec:
      requirements:
        - key: karpenter.k8s.aws/instance-family
          operator: In
          values: ["c5", "c6i", "c7g", "m5", "m6i", "m7g"]
          minValues: 3

what does minValues do and is my config correct?
```

**Drift issue (AMI rollout replacing all nodes at once):**
```
we updated our EC2NodeClass to use a new AMI version (changed amiSelectorTerms from v20240703 to v20240807). karpenter is now showing nodes as "Drifted" but it's replacing them all at once instead of doing a rolling replacement. we have 15 nodes and they're all getting replaced simultaneously, causing downtime.

apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: default
spec:
  amiSelectorTerms:
    - alias: al2023@v20240807
  role: KarpenterNodeRole
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: prod-cluster
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: prod-cluster

our NodePool disruption budget is set to nodes: "50%" but it doesn't seem to be limiting the drift replacements. is drift not subject to disruption budgets?
```

### 2. Low confidence (LLM fallback)

```
I forwarded this from my manager, can you take a look?
```

```
hey can someone help me with my cluster
```

### 3. Noise (rejected immediately)

```
what is the weather like today in Seattle?
```

## Setup (one-time)

```bash
# Full setup: OpenSearch, knowledge base, orchestrator image, Pod Identity
# Requires an arm64 build host for the container image
ARM64_BUILD_HOST=my-graviton-host ./scripts/setup-demo.sh

# Or if running on an arm64 machine with Docker:
./scripts/setup-demo.sh
```

The setup script:
1. Verifies SLM models are deployed
2. Deploys OpenSearch (if not already running)
3. Builds and pushes the orchestrator + embedding + reranker images (multi-arch)
4. Deploys the embedding, reranker, and orchestrator pods (Pod Identity for Bedrock LLM fallback)
5. Indexes the knowledge base (~3900 chunks from Karpenter, KEDA, EKS docs + blogs) using the in-cluster embedding service
6. Waits for everything to be ready

## Running Locally

```bash
# Prerequisites: models deployed via slemify deploy
# Port-forwards:
kubectl port-forward -n slemify svc/k8s-autoscaling-triage-inference 8081:8080
kubectl port-forward -n slemify svc/k8s-autoscaling-auditor-inference 8082:8080
kubectl port-forward -n slemify svc/opensearch-cluster-master 9200:9200
kubectl port-forward -n slemify svc/k8s-autoscaling-embedding 8083:80
kubectl port-forward -n slemify svc/k8s-autoscaling-reranker 8084:80

# Install deps
pip install fastapi uvicorn httpx opensearch-py boto3

# Start server (warms up models on boot)
python3 server.py

# Open http://localhost:8000
```

## Deploying to Cluster

```bash
# One command: builds images, pushes to ECR, sets up Pod Identity, deploys
./scripts/deploy.sh

# Access via port-forward
kubectl port-forward -n slemify svc/k8s-autoscaling-orchestrator 8000:80
# Open http://localhost:8000
```

The deploy script:
1. Builds the orchestrator + embedding + reranker images and pushes to ECR
2. Creates the ServiceAccount, Deployments, and Services
3. Sets up an IAM role with Bedrock access (LLM fallback) via EKS Pod Identity
4. Waits for the rollout to complete

## Live Demo

```bash
# Show infrastructure (CPU-only nodes, pods, architecture) in a separate terminal
./scripts/show-infra.sh

# Launch full demo: port-forward + browser + tmux log dashboard
./scripts/demo-terminal.sh
```

## Indexing the Knowledge Base

Indexing uses the in-cluster embedding service, so port-forward both OpenSearch
and the embedding pod first.

```bash
# Full index (clones repos, chunks, embeds, indexes)
kubectl port-forward -n slemify svc/opensearch-cluster-master 9200:9200
kubectl port-forward -n slemify svc/k8s-autoscaling-embedding 8083:80
pip install opensearch-py httpx gitpython requests beautifulsoup4
python3 scripts/index-knowledge.py

# Add only blog posts to existing index
python3 scripts/index-knowledge.py --append --source=aws-blog

# Add only a specific source
python3 scripts/index-knowledge.py --append --source=karpenter
```

> Note: the embedding model (bge-base-en-v1.5, 768d) must match at index and
> query time. The index (`k8s-autoscaling-knowledge`) is built with the
> in-cluster embedding service. Changing the embedding model requires a full
> reindex (the index mapping dimension changes), not an `--append`.

## The Story This Tells

1. CPUs handle the full AI pipeline (triage + embedding + reranking + retrieval + generation)
2. Fine-tuned SLMs are more accurate than general LLMs on domain-specific tasks
3. RAG grounds the response in current documentation (reduces hallucinations)
4. The tiered architecture (SLM router + SLM expert + LLM fallback) optimizes cost
5. Kubernetes-native: Karpenter provisions nodes, KEDA scales, OpenSearch runs in-cluster
6. Everything on Spot instances with consolidation
