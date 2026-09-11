// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"fmt"

	"github.com/aws-samples/sample-slemify/pkg/config"
)

// Provisioner is who owns the nodes: self-managed Karpenter, or EKS Auto Mode.
// Both speak karpenter.sh/v1 NodePool, but they differ in two places that
// matter here: the NodeClass kind the pool references and the group prefix of
// the well-known instance labels.
type Provisioner string

const (
	// ProvisionerKarpenter: Karpenter installed by you. Slemify creates its own
	// EC2NodeClass (Bottlerocket + SOCI).
	ProvisionerKarpenter Provisioner = "karpenter"
	// ProvisionerAutoMode: EKS Auto Mode. AWS owns the NodeClass, the AMI, and
	// the node lifecycle. Slemify only creates a NodePool that references the
	// cluster's existing eks.amazonaws.com NodeClass. No EC2NodeClass, no
	// userData.
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

// Instance generations per pool. Newer generations carry more memory
// bandwidth per socket, which is what CPU inference speed is made of, so the
// preferred pool is the newest generation and the fallback is the two before
// it. Older than that is not eligible at all.
const (
	preferredGeneration = `"8"`
	fallbackGenerations = `"6", "7"`
)

// SLMNodePoolManifests generates the CPU NodePools for all Slemify workloads
// (convert/train jobs and inference serving) as two YAML documents:
//
//   - slemify-slm (weight 100): current generation only.
//   - slemify-slm-fallback (weight 50): the two previous generations.
//
// The provisioner tries pools in weight order and falls through when the
// preferred one cannot launch (no capacity for that generation in the zone),
// so preference is expressed without an alpha feature and works the same on
// self-managed Karpenter and EKS Auto Mode. Both pools: c, m, and r families,
// arm64 and amd64, on-demand, the same slemify.io/workload label and
// slemify.io/slm taint, so workloads never know which pool served them.
func SLMNodePoolManifests(sized config.SizedConfig, opts NodePoolOptions) string {
	return nodePool("slemify-slm", 100, preferredGeneration, opts) +
		"---\n" +
		nodePool("slemify-slm-fallback", 50, fallbackGenerations, opts)
}

func nodePool(name string, weight int, generations string, opts NodePoolOptions) string {
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
  name: %s
  labels:
    app.kubernetes.io/managed-by: slemify
spec:
  weight: %d
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
          operator: In
          values: [%s]
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
`, name, weight, ncGroup, ncKind, ncName, labelGroup, categories, labelGroup, generations, labelGroup)
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
