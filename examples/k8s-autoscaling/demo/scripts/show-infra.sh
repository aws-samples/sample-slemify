#!/bin/bash
# Show the infrastructure running the demo in a live-updating display.
# Run this in a separate terminal/screen during the demo.

watch -n 5 '
echo "=== Nodes running SLM workloads (CPU only, no GPU) ==="
echo ""
kubectl get nodes -l slemify.io/workload=slm \
  -o custom-columns="\
NODE:.metadata.name,\
INSTANCE-TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,\
ARCH:.metadata.labels.kubernetes\.io/arch,\
vCPU:.status.capacity.cpu,\
MEMORY:.status.capacity.memory,\
GPU:.status.capacity.nvidia\.com/gpu"
echo ""

echo "=== Pods in slemify namespace ==="
echo ""
kubectl get pods -n slemify -l "app in (k8s-autoscaling-orchestrator,k8s-autoscaling-triage-inference,k8s-autoscaling-auditor-inference,k8s-autoscaling-embedding,opensearch-cluster-master)" \
  -o custom-columns="\
NAME:.metadata.name,\
STATUS:.status.phase,\
NODE:.spec.nodeName,\
CPU-REQ:.spec.containers[0].resources.requests.cpu,\
MEM-REQ:.spec.containers[0].resources.requests.memory"
echo ""

echo "=== Architecture ==="
echo ""
echo "  [Chat UI] --> [Orchestrator Pod]"
echo "                    |"
echo "                    ├── Triage SLM (4B, CPU, ~1.5s)"
echo "                    |       classifies intent + confidence"
echo "                    |"
echo "                    ├── OpenSearch (Vector DB, CPU, ~100ms)"
echo "                    |       retrieves relevant K8s docs"
echo "                    |"
echo "                    └── Auditor SLM (8B, CPU, streaming)"
echo "                            structured config analysis"
echo ""
echo "  All inference on CPU instances. Zero GPUs."
'
