// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package training

import (
	"strings"
	"testing"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

func karpenterExpertConfig() *config.ExpertConfig {
	return &config.ExpertConfig{
		APIVersion: "slemify/v1",
		Project: config.ProjectConfig{
			Name:          "karpenter-expert",
			Domain:        "Karpenter configuration and optimization on EKS",
			DomainVersion: "1.2",
			Task:          config.TaskGeneration,
		},
		Model: config.ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		Data: config.DataConfig{
			Bucket: "my-bucket",
			Path:   "karpenter-data/",
			Sources: []config.SourceConfig{
				{Path: "github-issues/", Type: "github-issues"},
				{Path: "docs/", Type: "documentation"},
			},
			Synthetic: config.SyntheticConfig{
				Model: "claude-sonnet",
				Pairs: 500,
			},
		},
		Training: config.TrainingConfig{Spot: true},
	}
}

func TestTrainingJobManifestLabels(t *testing.T) {
	cfg := karpenterExpertConfig()
	sized := config.AutoSize(cfg.Model, cfg.Data, cfg.Training)
	job := TrainingJobManifest(cfg, sized, "slemify", "cm", pipeline.NewPipelineContext())

	if job.Name != "karpenter-expert-training" {
		t.Errorf("job name = %q, want %q", job.Name, "karpenter-expert-training")
	}
	if job.Labels["slemify.io/stage"] != "training" {
		t.Errorf("stage label = %q, want %q", job.Labels["slemify.io/stage"], "training")
	}
}

func TestTrainingJobGPUResourceLimit(t *testing.T) {
	cfg := karpenterExpertConfig()
	sized := config.AutoSize(cfg.Model, cfg.Data, cfg.Training)
	job := TrainingJobManifest(cfg, sized, "slemify", "cm", pipeline.NewPipelineContext())

	c := job.Spec.Template.Spec.Containers[0]
	gpuLimit, ok := c.Resources.Limits["nvidia.com/gpu"]
	if !ok {
		t.Fatal("training container should have nvidia.com/gpu resource limit")
	}
	if gpuLimit.String() != "1" {
		t.Errorf("GPU limit = %s, want 1", gpuLimit.String())
	}
}

func TestTrainingJobSpotNodeSelector(t *testing.T) {
	cfg := karpenterExpertConfig()
	cfg.Training.Spot = true
	sized := config.AutoSize(cfg.Model, cfg.Data, cfg.Training)
	job := TrainingJobManifest(cfg, sized, "slemify", "cm", pipeline.NewPipelineContext())

	ns := job.Spec.Template.Spec.NodeSelector
	if _, hasCapacity := ns["karpenter.sh/capacity-type"]; hasCapacity {
		t.Error("pod should not set karpenter.sh/capacity-type (NodePool controls this)")
	}
	if ns["kubernetes.io/arch"] != "amd64" {
		t.Errorf("arch = %q, want 'amd64'", ns["kubernetes.io/arch"])
	}
}

func TestTrainingJobOnDemandNodeSelector(t *testing.T) {
	cfg := karpenterExpertConfig()
	cfg.Training.Spot = false
	sized := config.AutoSize(cfg.Model, cfg.Data, cfg.Training)
	job := TrainingJobManifest(cfg, sized, "slemify", "cm", pipeline.NewPipelineContext())

	ns := job.Spec.Template.Spec.NodeSelector
	if _, hasCapacity := ns["karpenter.sh/capacity-type"]; hasCapacity {
		t.Error("on-demand training should not set karpenter.sh/capacity-type")
	}
}

func TestTrainingJobGPUToleration(t *testing.T) {
	cfg := karpenterExpertConfig()
	sized := config.AutoSize(cfg.Model, cfg.Data, cfg.Training)
	job := TrainingJobManifest(cfg, sized, "slemify", "cm", pipeline.NewPipelineContext())

	tolerations := job.Spec.Template.Spec.Tolerations
	found := false
	for _, tol := range tolerations {
		if tol.Key == "nvidia.com/gpu" {
			found = true
			break
		}
	}
	if !found {
		t.Error("training job should tolerate nvidia.com/gpu taint")
	}
}

func TestTrainingJobBackoffLimit(t *testing.T) {
	cfg := karpenterExpertConfig()
	sized := config.AutoSize(cfg.Model, cfg.Data, cfg.Training)
	job := TrainingJobManifest(cfg, sized, "slemify", "cm", pipeline.NewPipelineContext())

	if *job.Spec.BackoffLimit != 6 {
		t.Errorf("backoffLimit = %d, want 6 (Spot recovery + image pull retries)", *job.Spec.BackoffLimit)
	}
}

func TestTrainingJobS3CheckpointEnv(t *testing.T) {
	cfg := karpenterExpertConfig()
	sized := config.AutoSize(cfg.Model, cfg.Data, cfg.Training)
	job := TrainingJobManifest(cfg, sized, "slemify", "cm", pipeline.NewPipelineContext())

	c := job.Spec.Template.Spec.Containers[0]
	found := false
	for _, env := range c.Env {
		if env.Name == "S3_CHECKPOINT_PATH" {
			if !strings.Contains(env.Value, "my-bucket") || !strings.Contains(env.Value, "karpenter-expert") {
				t.Errorf("S3_CHECKPOINT_PATH = %q, should contain bucket and project", env.Value)
			}
			found = true
			break
		}
	}
	if !found {
		t.Error("training container should have S3_CHECKPOINT_PATH env var")
	}
}
