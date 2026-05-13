# Slemify

Domain-specific AI experts on Kubernetes, from a single YAML config.

## Quick Start

```bash
# Generate a starter config
slemify init

# Validate config and see auto-sized values
slemify validate --config expert.yaml

# Run the full pipeline
slemify deploy --config expert.yaml

# Check pipeline status
slemify status --config expert.yaml

# Stream logs from a stage
slemify logs --stage training --follow

# Run inference cost analysis
slemify analyze --config expert.yaml
```

## Project Structure

```
cmd/           CLI commands (cobra)
pkg/config/    Schema, validator, auto-sizer
pkg/pipeline/  Stage sequencer, state machine
pkg/k8s/       client-go wrapper, manifest apply
pkg/data/      Synthetic data generation stage
pkg/training/  Unsloth training script gen, Job submission
pkg/serving/   Deployment, NodePool, KEDA gen, report Job
pkg/report/    Report types, HTML template
pkg/build/     Multi-arch container builds via EC2
containers/    Python containers (data-pipeline, training)
examples/      Reference implementations
```

## Building

```bash
# Build for current platform
go build -o slemify .

# Cross-compile
GOOS=linux GOARCH=amd64 go build -o slemify-linux-amd64 .
GOOS=linux GOARCH=arm64 go build -o slemify-linux-arm64 .
GOOS=darwin GOARCH=arm64 go build -o slemify-darwin-arm64 .
```

## Global Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `expert.yaml` | Path to Expert Config file |
| `--namespace` | `slemify` | Kubernetes namespace |
| `--kubeconfig` | `~/.kube/config` | Path to kubeconfig |
| `--image-registry` | (auto-detected) | Container image registry. Auto-detects ECR from the AWS account on EKS clusters. Override for custom registries. |

## Key Features

- **S3 mount for model serving.** If the Mountpoint for Amazon S3 CSI driver is installed, models are mounted directly from S3 via mmap (no download step). Falls back to init container download otherwise.
- **Memory locking.** The `--mlock` flag locks the model in RAM, preventing page eviction that degrades throughput on idle pods.
- **LLM-as-judge report.** The report stage uses Bedrock to semantically judge each prediction, with model confidence from token logprobs.
- **Data-driven serving config.** Context window and max_tokens are computed from actual training data output lengths, not hardcoded values.
