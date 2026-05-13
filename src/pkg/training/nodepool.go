// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package training

import (
	"fmt"
	"strings"

	"github.com/aws-samples/sample-slemify/pkg/config"
)

// GPUNodePoolManifest generates the shared Karpenter NodePool for GPU training workloads.
// Uses instance-category and GPU requirements for flexibility instead of pinning instance types.
func GPUNodePoolManifest(cfg *config.ExpertConfig, sized config.SizedConfig) string {
	capacityTypes := `"spot", "on-demand"`
	if !cfg.Training.Spot {
		capacityTypes = `"on-demand"`
	}

	// Single GPU for ≤10B models (tool's target range).
	// Karpenter picks the cheapest available instance with 1 NVIDIA GPU.
	// NodePool doesn't restrict GPU count — the pod requests what it needs.
	categories := `"g", "p"`
	if EstimateModelSize(cfg.Model.Base) > 13 {
		categories = `"p"`
	}

	return fmt.Sprintf(`apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: slemify-gpu
  labels:
    app.kubernetes.io/managed-by: slemify
spec:
  template:
    metadata:
      labels:
        slemify.io/workload: gpu
    spec:
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: slemify-gpu
      requirements:
        - key: karpenter.k8s.aws/instance-gpu-manufacturer
          operator: In
          values: ["nvidia"]
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values: [%s]
        - key: karpenter.k8s.aws/instance-generation
          operator: Gt
          values: ["4"]
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: [%s]
        - key: karpenter.k8s.aws/instance-size
          operator: NotIn
          values: ["metal"]
      taints:
        - key: nvidia.com/gpu
          effect: NoSchedule
  limits:
    cpu: "192"
    memory: "768Gi"
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 5m
`, categories, capacityTypes)
}

// GPUEC2NodeClassManifest generates the shared EC2NodeClass for GPU training.
// Uses Bottlerocket with SOCI snapshotter for faster container image pulls.
// Training images (Unsloth, CUDA) are multi-GB, so parallel pulls save significant time.
// 200Gi data volume for model weights and checkpoints.
func GPUEC2NodeClassManifest(clusterName, nodeRole, projectName string) string {
	return fmt.Sprintf(`apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: slemify-gpu
  labels:
    app.kubernetes.io/managed-by: slemify
spec:
  role: %s
  amiSelectorTerms:
    - alias: bottlerocket@latest
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: %s
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: %s
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 4Gi
        volumeType: gp3
        encrypted: true
        deleteOnTermination: true
    - deviceName: /dev/xvdb
      ebs:
        volumeSize: 200Gi
        volumeType: gp3
        iops: 6000
        throughput: 500
        encrypted: true
        deleteOnTermination: true
  userData: |
    [settings.container-runtime]
    snapshotter = "soci"

    [settings.container-runtime-plugins.soci-snapshotter]
    pull-mode = "parallel-pull-unpack"

    [settings.container-runtime-plugins.soci-snapshotter.parallel-pull-unpack]
    max-concurrent-downloads-per-image = 20
    concurrent-download-chunk-size = "16mb"
    max-concurrent-unpacks-per-image = 12
    discard-unpacked-layers = true
  tags:
    app.kubernetes.io/managed-by: slemify
    slemify.io/project: %s
    slemify.io/workload: training
`, nodeRole, clusterName, clusterName, projectName)
}

// EstimateModelSize returns approximate parameter count in billions.
func EstimateModelSize(modelID string) int {
	lower := strings.ToLower(modelID)
	hints := []struct {
		pattern string
		size    int
	}{
		{"70b", 70}, {"14b", 14}, {"13b", 13}, {"8b", 8},
		{"7b", 7}, {"3b", 3}, {"2b", 2}, {"1b", 1},
	}
	for _, h := range hints {
		if strings.Contains(lower, h.pattern) {
			return h.size
		}
	}
	return 7
}
