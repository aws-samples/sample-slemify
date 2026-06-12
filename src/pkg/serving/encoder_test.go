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

func TestEncoderServiceURL(t *testing.T) {
	got := EncoderServiceURL("triage", "slemify")
	want := "http://triage-encoder.slemify.svc.cluster.local:8080"
	if got != want {
		t.Errorf("EncoderServiceURL = %q, want %q", got, want)
	}
}

func TestGenerateEncoderManifests(t *testing.T) {
	cfg := classifierConfig()
	m := GenerateEncoderManifests(cfg, "slemify", pipeline.NewPipelineContext())

	if m.Deployment.Name != "k8s-autoscaling-triage-encoder" {
		t.Errorf("deployment name = %q", m.Deployment.Name)
	}
	c := m.Deployment.Spec.Template.Spec.Containers[0]

	// Encoder must be a Slemify-built image and serve the configured base model
	// via the EMBEDDING_MODEL_NAME env var (downloaded at startup).
	var foundModel bool
	for _, e := range c.Env {
		if e.Name == "EMBEDDING_MODEL_NAME" && e.Value == "BAAI/bge-base-en-v1.5" {
			foundModel = true
		}
	}
	if !foundModel {
		t.Errorf("encoder env missing EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5: %v", c.Env)
	}

	// Security: must have a non-nil SecurityContext with no priv escalation.
	if c.SecurityContext == nil || c.SecurityContext.AllowPrivilegeEscalation == nil || *c.SecurityContext.AllowPrivilegeEscalation {
		t.Error("encoder container must set AllowPrivilegeEscalation=false")
	}

	// Resource limits must be set.
	if c.Resources.Limits.Memory().IsZero() {
		t.Error("encoder container must set a memory limit")
	}
}
