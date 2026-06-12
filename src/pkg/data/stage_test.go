// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package data

import (
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

func TestDataJobManifestLabels(t *testing.T) {
	cfg := karpenterExpertConfig()
	job := DataJobManifest(cfg, "slemify", "karpenter-expert-expert-config", pipeline.NewPipelineContext())

	if job.Name != "karpenter-expert-data" {
		t.Errorf("job name = %q, want %q", job.Name, "karpenter-expert-data")
	}
	if job.Namespace != "slemify" {
		t.Errorf("namespace = %q, want %q", job.Namespace, "slemify")
	}
	if job.Labels["slemify.io/project"] != "karpenter-expert" {
		t.Errorf("project label = %q, want %q", job.Labels["slemify.io/project"], "karpenter-expert")
	}
	if job.Labels["slemify.io/stage"] != "data" {
		t.Errorf("stage label = %q, want %q", job.Labels["slemify.io/stage"], "data")
	}
	if job.Labels["app.kubernetes.io/managed-by"] != "slemify" {
		t.Errorf("managed-by label = %q, want %q", job.Labels["app.kubernetes.io/managed-by"], "slemify")
	}
}

func TestDataJobManifestConfigMapVolume(t *testing.T) {
	cfg := karpenterExpertConfig()
	cmName := "karpenter-expert-expert-config"
	job := DataJobManifest(cfg, "slemify", cmName, pipeline.NewPipelineContext())

	spec := job.Spec.Template.Spec
	if len(spec.Volumes) != 2 {
		t.Fatalf("volumes count = %d, want 2 (config + tmp)", len(spec.Volumes))
	}
	vol := spec.Volumes[0]
	if vol.ConfigMap == nil {
		t.Fatal("volume should be a ConfigMap")
	}
	if vol.ConfigMap.Name != cmName {
		t.Errorf("configmap name = %q, want %q", vol.ConfigMap.Name, cmName)
	}
}

func TestDataJobManifestContainer(t *testing.T) {
	cfg := karpenterExpertConfig()
	job := DataJobManifest(cfg, "slemify", "cm", pipeline.NewPipelineContext())

	containers := job.Spec.Template.Spec.Containers
	if len(containers) != 1 {
		t.Fatalf("containers count = %d, want 1", len(containers))
	}
	c := containers[0]
	if c.Image != pipeline.NewPipelineContext().Image("data-pipeline") {
		t.Errorf("image = %q, want data-pipeline image", c.Image)
	}
	foundConfig := false
	for _, vm := range c.VolumeMounts {
		if vm.MountPath == "/config" && vm.ReadOnly {
			foundConfig = true
		}
	}
	if !foundConfig {
		t.Error("container should mount config at /config (read-only)")
	}
}

func TestDataJobBackoffLimit(t *testing.T) {
	cfg := karpenterExpertConfig()
	job := DataJobManifest(cfg, "slemify", "cm", pipeline.NewPipelineContext())

	if *job.Spec.BackoffLimit != 2 {
		t.Errorf("backoffLimit = %d, want 2", *job.Spec.BackoffLimit)
	}
}
