// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"fmt"

	"github.com/aws-samples/sample-slemify/pkg/config"
)

// SLMNodePoolManifest generates the shared Karpenter NodePool for all CPU workloads:
// quantize and inference serving. Uses arm64 Graviton instances.
func SLMNodePoolManifest(sized config.SizedConfig) string {
	// Allow c, m, and r families for CPU inference — Karpenter picks the cheapest.
	categories := `"c", "m", "r"`

	return fmt.Sprintf(`apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: slemify-slm
  labels:
    app.kubernetes.io/managed-by: slemify
spec:
  template:
    metadata:
      labels:
        slemify.io/workload: slm
    spec:
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: slemify-slm
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["arm64", "amd64"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values: [%s]
        - key: karpenter.k8s.aws/instance-generation
          operator: Gt
          values: ["4"]
        - key: karpenter.k8s.aws/instance-size
          operator: NotIn
          values: ["metal", "nano", "micro", "small"]
        - key: slemify.io/workload
          operator: In
          values: ["slm"]
      taints:
        - key: slemify.io/slm
          effect: NoSchedule
  limits:
    cpu: "256"
    memory: "512Gi"
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 5m
`, categories)
}

// SLMEC2NodeClassManifest generates the shared EC2NodeClass for all CPU workloads.
// Uses Bottlerocket with SOCI snapshotter for faster container image pulls.
// Bottlerocket has native SOCI support, so no shell-based installation is needed.
func SLMEC2NodeClassManifest(clusterName, nodeRole, projectName string) string {
	return fmt.Sprintf(`apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: slemify-slm
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
        volumeSize: 80Gi
        volumeType: gp3
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
    slemify.io/workload: slm
`, nodeRole, clusterName, clusterName, projectName)
}



// NodeOverlayManifests generates Karpenter NodeOverlay resources that penalize
// older instance generations to prefer the latest (gen 8 Graviton/x86).
// Targeted to the slemify-slm NodePool so training GPU nodes are unaffected.
// With on-demand capacity, this gives deterministic latest-gen selection.
// With Spot, EC2 Fleet uses capacity-optimized-prioritized which may override
// preferences based on capacity availability.
func NodeOverlayManifests() string {
	return `apiVersion: karpenter.sh/v1alpha1
kind: NodeOverlay
metadata:
  name: slemify-penalize-gen5
  labels:
    app.kubernetes.io/managed-by: slemify
spec:
  weight: 10
  requirements:
    - key: karpenter.sh/nodepool
      operator: In
      values: ["slemify-slm"]
    - key: karpenter.k8s.aws/instance-generation
      operator: In
      values: ["5"]
  priceAdjustment: "+45%"
---
apiVersion: karpenter.sh/v1alpha1
kind: NodeOverlay
metadata:
  name: slemify-penalize-gen6
  labels:
    app.kubernetes.io/managed-by: slemify
spec:
  weight: 10
  requirements:
    - key: karpenter.sh/nodepool
      operator: In
      values: ["slemify-slm"]
    - key: karpenter.k8s.aws/instance-generation
      operator: In
      values: ["6"]
  priceAdjustment: "+30%"
---
apiVersion: karpenter.sh/v1alpha1
kind: NodeOverlay
metadata:
  name: slemify-penalize-gen7
  labels:
    app.kubernetes.io/managed-by: slemify
spec:
  weight: 10
  requirements:
    - key: karpenter.sh/nodepool
      operator: In
      values: ["slemify-slm"]
    - key: karpenter.k8s.aws/instance-generation
      operator: In
      values: ["7"]
  priceAdjustment: "+15%"
`
}

