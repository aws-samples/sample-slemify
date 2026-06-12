// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package training

import (
	"strings"
	"testing"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

func classifierExpertConfig() *config.ExpertConfig {
	return &config.ExpertConfig{
		APIVersion: "slemify/v1",
		Project: config.ProjectConfig{
			Name:   "k8s-autoscaling-triage",
			Domain: "Classify k8s autoscaling queries",
			Task:   config.TaskClassification,
			Labels: map[string][]string{"routing": {"a", "b"}},
		},
		Model: config.ModelConfig{Base: "BAAI/bge-base-en-v1.5", Head: "logistic"},
		Data:  config.DataConfig{Bucket: "my-bucket", Path: "data/", Synthetic: config.SyntheticConfig{Model: "claude", Pairs: 500}},
	}
}

func TestClassifierJobIsCPUOnly(t *testing.T) {
	cfg := classifierExpertConfig()
	job := ClassifierJobManifest(cfg, "slemify", pipeline.NewPipelineContext())

	c := job.Spec.Template.Spec.Containers[0]

	// Must NOT request a GPU.
	if _, ok := c.Resources.Limits["nvidia.com/gpu"]; ok {
		t.Error("classifier training job must not request a GPU")
	}

	// Must run on the SLM (CPU) node pool.
	if job.Spec.Template.Spec.NodeSelector["slemify.io/workload"] != "slm" {
		t.Errorf("classifier job should target slm workload, got %v", job.Spec.Template.Spec.NodeSelector)
	}

	// Must run the classifier training script.
	if strings.Join(c.Command, " ") != "python3 classifier_train.py" {
		t.Errorf("unexpected command: %v", c.Command)
	}

	// Must pass the encoder URL and head via env.
	var hasEncoder, hasHead bool
	for _, e := range c.Env {
		if e.Name == "ENCODER_URL" && strings.Contains(e.Value, "-encoder.") {
			hasEncoder = true
		}
		if e.Name == "HEAD" && e.Value == "logistic" {
			hasHead = true
		}
	}
	if !hasEncoder {
		t.Error("classifier job missing ENCODER_URL env")
	}
	if !hasHead {
		t.Error("classifier job missing HEAD env")
	}

	// Resource limits must be set.
	if c.Resources.Limits.Memory().IsZero() {
		t.Error("classifier container must set a memory limit")
	}
}
