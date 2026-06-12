// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"testing"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

func classifierConfig() *config.ExpertConfig {
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

func TestClassifierServingIsCPUClassifierMode(t *testing.T) {
	cfg := classifierConfig()
	sized := config.AutoSizeForTask(cfg.Model, cfg.Data, cfg.Training, cfg.Project.Task)
	m := GenerateClassifierInferenceManifests(cfg, sized, "slemify", pipeline.NewPipelineContext())

	c := m.Deployment.Spec.Template.Spec.Containers[0]

	// No GPU.
	if _, ok := c.Resources.Limits["nvidia.com/gpu"]; ok {
		t.Error("classifier serving must not request a GPU")
	}

	// Classifier serving (ONNX) loads artifacts from S3 — needs S3_BUCKET + PROJECT.
	env := map[string]string{}
	for _, e := range c.Env {
		env[e.Name] = e.Value
	}
	if env["S3_BUCKET"] != "my-bucket" || env["PROJECT"] != "k8s-autoscaling-triage" {
		t.Errorf("classifier serving missing S3_BUCKET/PROJECT env: %v", env)
	}
	// Serving is the lean ONNX image (no torch / sentence-transformers).
	if c.Image == "" {
		t.Error("classifier serving image not set")
	}

	// Security + limits.
	if c.SecurityContext == nil || c.SecurityContext.AllowPrivilegeEscalation == nil || *c.SecurityContext.AllowPrivilegeEscalation {
		t.Error("classifier container must set AllowPrivilegeEscalation=false")
	}
	if c.Resources.Limits.Memory().IsZero() {
		t.Error("classifier container must set a memory limit")
	}

	// Same service name as the generative path (drop-in swap).
	if m.Service.Name != "k8s-autoscaling-triage-inference" {
		t.Errorf("service name = %q, want k8s-autoscaling-triage-inference", m.Service.Name)
	}
}
