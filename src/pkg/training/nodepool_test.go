// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package training

import (
	"strings"
	"testing"

	"github.com/aws-samples/sample-slemify/pkg/config"
)

func karpenterConfig() *config.ExpertConfig {
	return &config.ExpertConfig{
		APIVersion: "slemify/v1",
		Project:    config.ProjectConfig{Name: "karpenter-expert", Domain: "Karpenter"},
		Model:      config.ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		Data:       config.DataConfig{Bucket: "test-bucket", Path: "test-data/", Synthetic: config.SyntheticConfig{Model: "claude", Pairs: 1000}},
		Training:   config.TrainingConfig{Spot: true},
	}
}

func sized7B() config.SizedConfig {
	return config.AutoSize(
		config.ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		config.DataConfig{Bucket: "b", Path: "p", Synthetic: config.SyntheticConfig{Model: "m", Pairs: 1000}},
		config.TrainingConfig{Spot: true},
	)
}

func TestGPUNodePoolSpot(t *testing.T) {
	cfg := karpenterConfig()
	cfg.Training.Spot = true
	sized := sized7B()

	manifest := GPUNodePoolManifest(cfg, sized)

	if !strings.Contains(manifest, "name: slemify-gpu") {
		t.Error("should be named slemify-gpu")
	}
	if !strings.Contains(manifest, `"amd64"`) {
		t.Error("should require amd64 architecture")
	}
	if !strings.Contains(manifest, `"spot"`) {
		t.Error("should include spot capacity when training.spot=true")
	}
	if !strings.Contains(manifest, `"on-demand"`) {
		t.Error("should include on-demand as fallback")
	}
	if !strings.Contains(manifest, "nvidia.com/gpu") {
		t.Error("should have GPU taint")
	}
	if !strings.Contains(manifest, `slemify.io/workload`) {
		t.Error("should have workload requirement for scheduling")
	}
}

func TestGPUNodePoolOnDemand(t *testing.T) {
	cfg := karpenterConfig()
	cfg.Training.Spot = false
	sized := sized7B()

	manifest := GPUNodePoolManifest(cfg, sized)

	if !strings.Contains(manifest, `"on-demand"`) {
		t.Error("should use on-demand capacity")
	}
	if strings.Contains(manifest, `"spot"`) {
		t.Error("should not contain spot when training.spot=false")
	}
}

func TestGPUNodePoolGPURequirements(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()

	manifest := GPUNodePoolManifest(cfg, sized)

	if !strings.Contains(manifest, "instance-gpu-manufacturer") {
		t.Error("should use GPU manufacturer requirement")
	}
	if !strings.Contains(manifest, `"nvidia"`) {
		t.Error("should require nvidia GPUs")
	}
	// NodePool should NOT restrict GPU count — pod expresses what it needs
	if strings.Contains(manifest, "instance-gpu-count") {
		t.Error("should not restrict GPU count in NodePool")
	}
	if !strings.Contains(manifest, `"g"`) || !strings.Contains(manifest, `"p"`) {
		t.Error("7B model should include g and p families")
	}
}

func TestGPUNodePoolExcludesMetal(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()

	manifest := GPUNodePoolManifest(cfg, sized)

	if !strings.Contains(manifest, `"metal"`) {
		t.Error("should exclude metal instances")
	}
}

func TestGPUEC2NodeClass(t *testing.T) {
	manifest := GPUEC2NodeClassManifest("my-cluster", "KarpenterNodeRole-my-cluster", "test-project")

	if !strings.Contains(manifest, "name: slemify-gpu") {
		t.Error("should be named slemify-gpu")
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
	if !strings.Contains(manifest, "200Gi") {
		t.Error("should have 200Gi volume for model weights and GGUF export")
	}
	if !strings.Contains(manifest, "KarpenterNodeRole-my-cluster") {
		t.Error("should reference the node role")
	}
}
