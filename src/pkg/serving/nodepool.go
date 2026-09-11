// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"fmt"

	"github.com/aws-samples/sample-slemify/pkg/config"
)

// Provisioner is who owns the nodes: self-managed Karpenter, or EKS Auto Mode.
// Both speak karpenter.sh/v1 NodePool, but they differ in three places that
// matter here: the NodeClass kind the pool references, the group prefix of the
// well-known instance labels, and whether NodeOverlay exists at all.
type Provisioner string

const (
	// ProvisionerKarpenter: Karpenter installed by you. Slemify creates its own
	// EC2NodeClass (Bottlerocket + SOCI) and applies NodeOverlays if enabled.
	ProvisionerKarpenter Provisioner = "karpenter"
	// ProvisionerAutoMode: EKS Auto Mode. AWS owns the NodeClass, the AMI, and
	// the node lifecycle. Slemify only creates a NodePool that references the
	// cluster's existing eks.amazonaws.com NodeClass. No EC2NodeClass, no
	// userData, no NodeOverlay.
	ProvisionerAutoMode Provisioner = "auto-mode"
)

// NodePoolOptions parameterize the shared CPU NodePool.
type NodePoolOptions struct {
	Provisioner Provisioner
	// NodeClassName is the NodeClass the pool references. Ignored for Karpenter
	// (always slemify-slm). For Auto Mode it is an existing NodeClass, normally
	// "default".
	NodeClassName string
}

// SLMNodePoolManifest generates the shared NodePool for all CPU workloads:
// convert/train jobs and inference serving. c, m, and r families, generation 5
// and newer, arm64 and amd64, on-demand: the provisioner picks the cheapest
// instance that fits. Requirement keys follow the provisioner's label group.
func SLMNodePoolManifest(sized config.SizedConfig, opts NodePoolOptions) string {
	categories := `"c", "m", "r"`
	labelGroup, ncGroup, ncKind, ncName := "karpenter.k8s.aws", "karpenter.k8s.aws", "EC2NodeClass", "slemify-slm"
	if opts.Provisioner == ProvisionerAutoMode {
		labelGroup, ncGroup, ncKind = "eks.amazonaws.com", "eks.amazonaws.com", "NodeClass"
		ncName = opts.NodeClassName
		if ncName == "" {
			ncName = "default"
		}
	}

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
        group: %s
        kind: %s
        name: %s
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["arm64", "amd64"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]
        - key: %s/instance-category
          operator: In
          values: [%s]
        - key: %s/instance-generation
          operator: Gt
          values: ["4"]
        - key: %s/instance-size
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
`, ncGroup, ncKind, ncName, labelGroup, categories, labelGroup, labelGroup)
}

// SLMEC2NodeClassManifest generates the shared EC2NodeClass for all CPU workloads
// on self-managed Karpenter. Uses Bottlerocket with SOCI snapshotter for faster
// container image pulls. Bottlerocket has native SOCI support, so no shell-based
// installation is needed. Not used on EKS Auto Mode, where AWS owns the NodeClass.
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
// preferences based on capacity availability. Self-managed Karpenter only: EKS
// Auto Mode has no NodeOverlay CRD, so there the pool's generation floor is the
// only lever and the provisioner picks the cheapest fit above it.
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
