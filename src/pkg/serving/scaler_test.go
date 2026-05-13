// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"fmt"
	"strings"
	"testing"
)

func TestKEDAScaledObjectTargetsDeployment(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	manifest := KEDAScaledObjectManifest(cfg, sized, "slemify")

	expected := "name: karpenter-expert-inference"
	if !strings.Contains(manifest, expected) {
		t.Errorf("ScaledObject should target deployment %q", expected)
	}
}

func TestKEDAScaledObjectReplicaCounts(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	manifest := KEDAScaledObjectManifest(cfg, sized, "slemify")

	if !strings.Contains(manifest, "minReplicaCount: 1") {
		t.Error("minReplicaCount should be 1")
	}
	expected := fmt.Sprintf("maxReplicaCount: %d", sized.KEDAMaxReplicas)
	if !strings.Contains(manifest, expected) {
		t.Errorf("maxReplicaCount should be %d", sized.KEDAMaxReplicas)
	}
}

func TestKEDAScaledObjectPrometheusMetric(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	manifest := KEDAScaledObjectManifest(cfg, sized, "slemify")

	if !strings.Contains(manifest, "llamacpp_requests_processing") {
		t.Error("should use llamacpp_requests_processing metric")
	}
	if !strings.Contains(manifest, "type: prometheus") {
		t.Error("trigger type should be prometheus")
	}
}

func TestKEDAScaledObjectLabels(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	manifest := KEDAScaledObjectManifest(cfg, sized, "slemify")

	if !strings.Contains(manifest, "slemify.io/project: karpenter-expert") {
		t.Error("should have project label")
	}
	if !strings.Contains(manifest, "namespace: slemify") {
		t.Error("should have correct namespace")
	}
}
