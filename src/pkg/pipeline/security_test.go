// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package pipeline_test

import (
	"strings"
	"testing"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/data"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
	"github.com/aws-samples/sample-slemify/pkg/serving"
	"github.com/aws-samples/sample-slemify/pkg/training"
	corev1 "k8s.io/api/core/v1"
)

func secConfig() *config.ExpertConfig {
	return &config.ExpertConfig{
		APIVersion: "slemify/v1",
		Project:    config.ProjectConfig{Name: "test-project", Domain: "test domain"},
		Model:      config.ModelConfig{Base: "HuggingFaceTB/SmolLM3-3B"},
		Data: config.DataConfig{
			Bucket: "test-bucket",
			Path:   "test-data/",
			Synthetic: config.SyntheticConfig{
				Model: "anthropic.claude-3-haiku",
				Pairs: 100,
			},
		},
	}
}

func secSized() config.SizedConfig {
	c := secConfig()
	return config.AutoSize(c.Model, c.Data, c.Training)
}

// allContainers collects every container from every pod-generating function.
func allContainers() map[string]corev1.Container {
	cfg := secConfig()
	sized := secSized()
	ns := "test-ns"
	result := make(map[string]corev1.Container)

	dj := data.DataJobManifest(cfg, ns, "test-cm", pipeline.NewPipelineContext())
	for _, c := range dj.Spec.Template.Spec.InitContainers {
		result["data-init/"+c.Name] = c
	}
	for _, c := range dj.Spec.Template.Spec.Containers {
		result["data/"+c.Name] = c
	}

	tj := training.TrainingJobManifest(cfg, sized, ns, "test-cm", pipeline.NewPipelineContext())
	for _, c := range tj.Spec.Template.Spec.InitContainers {
		result["training-init/"+c.Name] = c
	}
	for _, c := range tj.Spec.Template.Spec.Containers {
		result["training/"+c.Name] = c
	}

	im := serving.GenerateInferenceManifests(cfg, sized, ns, pipeline.NewPipelineContext())
	for _, c := range im.Deployment.Spec.Template.Spec.InitContainers {
		result["serving-init/"+c.Name] = c
	}
	for _, c := range im.Deployment.Spec.Template.Spec.Containers {
		result["serving/"+c.Name] = c
	}

	return result
}

// --- Security Context ---

func TestSecurityAllContainersHaveSecurityContext(t *testing.T) {
	for name, c := range allContainers() {
		if c.SecurityContext == nil {
			t.Errorf("%s: missing SecurityContext", name)
		}
	}
}

func TestSecurityNoPrivilegeEscalation(t *testing.T) {
	for name, c := range allContainers() {
		if c.SecurityContext == nil {
			continue
		}
		if name == "training/training" {
			continue // runs as root for GPU access
		}
		if c.SecurityContext.AllowPrivilegeEscalation == nil || *c.SecurityContext.AllowPrivilegeEscalation {
			t.Errorf("%s: AllowPrivilegeEscalation must be false", name)
		}
	}
}

// --- Resources ---

func TestSecurityAllContainersHaveMemoryRequest(t *testing.T) {
	for name, c := range allContainers() {
		if c.Resources.Requests == nil {
			t.Errorf("%s: missing resource requests", name)
			continue
		}
		if _, ok := c.Resources.Requests[corev1.ResourceMemory]; !ok {
			t.Errorf("%s: missing memory request", name)
		}
	}
}

func TestSecurityMemoryRequestEqualsLimit(t *testing.T) {
	for name, c := range allContainers() {
		req := c.Resources.Requests[corev1.ResourceMemory]
		lim := c.Resources.Limits[corev1.ResourceMemory]
		if req.Cmp(lim) != 0 {
			t.Errorf("%s: memory request (%s) != limit (%s)", name, req.String(), lim.String())
		}
	}
}

func TestSecurityNoCPULimits(t *testing.T) {
	for name, c := range allContainers() {
		if c.Resources.Limits != nil {
			if _, has := c.Resources.Limits[corev1.ResourceCPU]; has {
				t.Errorf("%s: has CPU limit — only requests allowed", name)
			}
		}
	}
}

func TestSecurityMemoryLimitsPresent(t *testing.T) {
	for name, c := range allContainers() {
		if c.Resources.Limits == nil {
			t.Errorf("%s: missing resource limits", name)
			continue
		}
		if _, ok := c.Resources.Limits[corev1.ResourceMemory]; !ok {
			t.Errorf("%s: missing memory limit", name)
		}
	}
}

// --- Labels ---

func TestSecurityProjectLabel(t *testing.T) {
	cfg := secConfig()
	sized := secSized()
	ns := "test-ns"

	dj := data.DataJobManifest(cfg, ns, "test-cm", pipeline.NewPipelineContext())
	assertLbl(t, "data", dj.Labels, "slemify.io/project", cfg.Project.Name)

	tj := training.TrainingJobManifest(cfg, sized, ns, "test-cm", pipeline.NewPipelineContext())
	assertLbl(t, "training", tj.Labels, "slemify.io/project", cfg.Project.Name)

	im := serving.GenerateInferenceManifests(cfg, sized, ns, pipeline.NewPipelineContext())
	assertLbl(t, "serving", im.Deployment.Labels, "slemify.io/project", cfg.Project.Name)
}

func TestSecurityManagedByLabel(t *testing.T) {
	cfg := secConfig()
	sized := secSized()
	ns := "test-ns"

	dj := data.DataJobManifest(cfg, ns, "test-cm", pipeline.NewPipelineContext())
	assertLbl(t, "data", dj.Labels, "app.kubernetes.io/managed-by", "slemify")

	tj := training.TrainingJobManifest(cfg, sized, ns, "test-cm", pipeline.NewPipelineContext())
	assertLbl(t, "training", tj.Labels, "app.kubernetes.io/managed-by", "slemify")
}

// --- No Hardcoded Secrets ---

func TestSecurityNoHardcodedSecrets(t *testing.T) {
	patterns := []string{"AWS_ACCESS_KEY", "AWS_SECRET", "PASSWORD", "API_KEY"}
	for name, c := range allContainers() {
		for _, env := range c.Env {
			upper := strings.ToUpper(env.Name)
			for _, p := range patterns {
				if strings.Contains(upper, p) && env.ValueFrom == nil {
					t.Errorf("%s: env %s looks like a secret but has plain value", name, env.Name)
				}
			}
		}
	}
}

// --- Container Images ---

func TestSecurityNoUnexpectedLatestTag(t *testing.T) {
	allowed := map[string]bool{
		"unsloth/unsloth:2026.4.8-pt2.10.0-vllm-0.16.0-cu12.8-studio-release-v0.1.37-beta-fix-startup": true,
		"amazon/aws-cli:latest":  true,
	}
	for name, c := range allContainers() {
		if strings.HasSuffix(c.Image, ":latest") && !allowed[c.Image] && !strings.Contains(c.Image, "slemify/") {
			t.Errorf("%s: image %s uses :latest — pin a version", name, c.Image)
		}
	}
}

// --- Karpenter Annotations ---

func TestSecurityBatchJobsHaveDoNotDisrupt(t *testing.T) {
	cfg := secConfig()
	sized := secSized()
	ns := "test-ns"

	tj := training.TrainingJobManifest(cfg, sized, ns, "test-cm", pipeline.NewPipelineContext())
	if tj.Spec.Template.Annotations["karpenter.sh/do-not-disrupt"] != "true" {
		t.Error("training: missing do-not-disrupt annotation")
	}
}

// --- Input Validation ---

func TestSecurityS3PathNoTraversal(t *testing.T) {
	bad := &config.ExpertConfig{
		APIVersion: "slemify/v1",
		Project:    config.ProjectConfig{Name: "../../../etc", Domain: "test"},
		Model:      config.ModelConfig{Base: "test-3B"},
		Data:       config.DataConfig{Bucket: "b", Path: "p/", Synthetic: config.SyntheticConfig{Model: "m", Pairs: 100}},
	}
	errs := config.Validate(bad)
	found := false
	for _, e := range errs {
		if e.Field == "project.name" {
			found = true
		}
	}
	if !found {
		t.Error("validator should reject project name with path traversal")
	}
}

// --- Helpers ---

func assertLbl(t *testing.T, res string, labels map[string]string, key, want string) {
	t.Helper()
	if labels == nil {
		t.Errorf("%s: no labels", res)
		return
	}
	if got := labels[key]; got != want {
		t.Errorf("%s: label %s = %q, want %q", res, key, got, want)
	}
}
