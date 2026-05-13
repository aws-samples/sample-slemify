// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"strings"
	"testing"
)

func TestSLMNodePoolArchFlexible(t *testing.T) {
	sized := sized7B()
	manifest := SLMNodePoolManifest(sized)

	if !strings.Contains(manifest, `"arm64"`) {
		t.Error("SLM NodePool should allow arm64")
	}
	if !strings.Contains(manifest, `"amd64"`) {
		t.Error("SLM NodePool should allow amd64")
	}
}

func TestSLMNodePoolCapacityTypes(t *testing.T) {
	sized := sized7B()
	manifest := SLMNodePoolManifest(sized)

	if !strings.Contains(manifest, `"on-demand"`) {
		t.Error("SLM NodePool should use on-demand for deterministic NodeOverlay behavior")
	}
	if strings.Contains(manifest, `"spot"`) {
		t.Error("SLM NodePool should not include spot (NodeOverlay price adjustments need on-demand)")
	}
}

func TestSLMNodePoolInstanceCategory(t *testing.T) {
	sized := sized7B()
	manifest := SLMNodePoolManifest(sized)

	if !strings.Contains(manifest, "instance-category") {
		t.Error("should use instance-category for flexibility")
	}
	if !strings.Contains(manifest, `"c"`) || !strings.Contains(manifest, `"r"`) {
		t.Error("should allow c and r families")
	}
}

func TestSLMNodePoolExcludesSmallAndMetal(t *testing.T) {
	sized := sized7B()
	manifest := SLMNodePoolManifest(sized)

	if !strings.Contains(manifest, `"metal"`) {
		t.Error("should exclude metal instances")
	}
	if !strings.Contains(manifest, `"nano"`) {
		t.Error("should exclude nano instances")
	}
}

func TestSLMNodePoolTaint(t *testing.T) {
	sized := sized7B()
	manifest := SLMNodePoolManifest(sized)

	if !strings.Contains(manifest, "slemify.io/slm") {
		t.Error("should have slemify.io/slm taint")
	}
}

func TestSLMNodePoolName(t *testing.T) {
	sized := sized7B()
	manifest := SLMNodePoolManifest(sized)

	if !strings.Contains(manifest, "name: slemify-slm") {
		t.Error("should be named slemify-slm")
	}
}

func TestSLMNodePoolReferencesOwnNodeClass(t *testing.T) {
	sized := sized7B()
	manifest := SLMNodePoolManifest(sized)

	if !strings.Contains(manifest, "name: slemify-slm") {
		t.Error("should reference slemify-slm EC2NodeClass")
	}
}

func TestSLMNodePoolMinGeneration(t *testing.T) {
	sized := sized7B()
	manifest := SLMNodePoolManifest(sized)

	if !strings.Contains(manifest, "instance-generation") {
		t.Error("should filter by instance generation")
	}
	if !strings.Contains(manifest, `Gt`) {
		t.Error("should use Gt operator for generation")
	}
	if !strings.Contains(manifest, `"4"`) {
		t.Error("should require generation > 4 (allows gen 5+ for NodeOverlay penalization)")
	}
}

func TestSLMEC2NodeClass(t *testing.T) {
	manifest := SLMEC2NodeClassManifest("my-cluster", "KarpenterNodeRole-my-cluster", "test-project")

	if !strings.Contains(manifest, "name: slemify-slm") {
		t.Error("should be named slemify-slm")
	}
	if !strings.Contains(manifest, "bottlerocket@latest") {
		t.Error("should use Bottlerocket")
	}
	if !strings.Contains(manifest, "karpenter.sh/discovery: my-cluster") {
		t.Error("should use cluster name for discovery")
	}
	if !strings.Contains(manifest, "encrypted: true") {
		t.Error("EBS should be encrypted")
	}
	if !strings.Contains(manifest, "80Gi") {
		t.Error("should have 80Gi root volume for model processing")
	}
	if !strings.Contains(manifest, "KarpenterNodeRole-my-cluster") {
		t.Error("should reference the node role")
	}
}

func TestNodeOverlayManifests(t *testing.T) {
	manifest := NodeOverlayManifests()

	// Should have 3 overlays for gen 5, 6, 7
	if !strings.Contains(manifest, "penalize-gen5") {
		t.Error("should penalize generation 5")
	}
	if !strings.Contains(manifest, "penalize-gen6") {
		t.Error("should penalize generation 6")
	}
	if !strings.Contains(manifest, "penalize-gen7") {
		t.Error("should penalize generation 7")
	}

	// Should target slemify-slm NodePool
	if !strings.Contains(manifest, "slemify-slm") {
		t.Error("should target slemify-slm NodePool")
	}

	// Should have decreasing penalties (gen5 > gen6 > gen7)
	if !strings.Contains(manifest, `"+45%"`) {
		t.Error("gen5 should have +45% penalty")
	}
	if !strings.Contains(manifest, `"+30%"`) {
		t.Error("gen6 should have +30% penalty")
	}
	if !strings.Contains(manifest, `"+15%"`) {
		t.Error("gen7 should have +15% penalty")
	}

	// Gen 8 should have no overlay (no penalty = preferred)
	if strings.Contains(manifest, "gen8") {
		t.Error("gen8 should have no overlay (naturally preferred)")
	}
}
