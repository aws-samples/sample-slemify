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
		Project:    config.ProjectConfig{Name: "test-project", Domain: "test domain", Task: config.TaskGeneration},
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

	cj := training.ConvertJobManifest(cfg, sized, ns, pipeline.NewPipelineContext())
	for _, c := range cj.Spec.Template.Spec.InitContainers {
		result["training-init/"+c.Name] = c
	}
	for _, c := range cj.Spec.Template.Spec.Containers {
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

	cj := training.ConvertJobManifest(cfg, sized, ns, pipeline.NewPipelineContext())
	assertLbl(t, "training", cj.Labels, "slemify.io/project", cfg.Project.Name)

	im := serving.GenerateInferenceManifests(cfg, sized, ns, pipeline.NewPipelineContext())
	assertLbl(t, "serving", im.Deployment.Labels, "slemify.io/project", cfg.Project.Name)
}

func TestSecurityManagedByLabel(t *testing.T) {
	cfg := secConfig()
	sized := secSized()
	ns := "test-ns"

	dj := data.DataJobManifest(cfg, ns, "test-cm", pipeline.NewPipelineContext())
	assertLbl(t, "data", dj.Labels, "app.kubernetes.io/managed-by", "slemify")

	cj := training.ConvertJobManifest(cfg, sized, ns, pipeline.NewPipelineContext())
	assertLbl(t, "training", cj.Labels, "app.kubernetes.io/managed-by", "slemify")
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
	// Slemify-built images (ghcr.io/slemify/*) are tagged :latest by the build
	// pipeline; third-party images must be pinned.
	allowed := map[string]bool{}
	for name, c := range allContainers() {
		if strings.HasSuffix(c.Image, ":latest") && !allowed[c.Image] && !strings.Contains(c.Image, "slemify/") {
			t.Errorf("%s: image %s uses :latest — pin a version", name, c.Image)
		}
	}
}

// --- Convert Job (generation) ---

// TestSecurityConvertJobCPUOnly verifies the generation convert Job runs only on
// the CPU pool: no GPU nodeSelector, no nvidia toleration, no GPU resources.
func TestSecurityConvertJobCPUOnly(t *testing.T) {
	cfg := secConfig()
	sized := secSized()
	job := training.ConvertJobManifest(cfg, sized, "test-ns", pipeline.NewPipelineContext())
	spec := job.Spec.Template.Spec

	if spec.NodeSelector["slemify.io/workload"] != "slm" {
		t.Errorf("convert job must target the CPU (slm) pool, got nodeSelector %v", spec.NodeSelector)
	}
	for k := range spec.NodeSelector {
		if strings.Contains(k, "gpu") || strings.Contains(k, "nvidia") {
			t.Errorf("convert job must not have a GPU nodeSelector key, found %q", k)
		}
	}
	for _, tol := range spec.Tolerations {
		if strings.Contains(tol.Key, "nvidia") || strings.Contains(tol.Key, "gpu") {
			t.Errorf("convert job must not tolerate GPU taints, found %q", tol.Key)
		}
	}
	for _, c := range spec.Containers {
		if _, ok := c.Resources.Limits["nvidia.com/gpu"]; ok {
			t.Errorf("convert container %q must not request GPU resources", c.Name)
		}
	}
}

// TestSecurityConvertJobNoShellInjection verifies the convert Job exposes no
// shell-injection vector: it runs the image ENTRYPOINT with structured env vars
// rather than interpolating config values into a shell command string.
func TestSecurityConvertJobNoShellInjection(t *testing.T) {
	cfg := secConfig()
	sized := secSized()
	job := training.ConvertJobManifest(cfg, sized, "test-ns", pipeline.NewPipelineContext())
	for _, c := range job.Spec.Template.Spec.Containers {
		if len(c.Command) > 0 {
			t.Errorf("convert container %q should not override Command with a shell wrapper, got %v", c.Name, c.Command)
		}
		for _, arg := range c.Args {
			if strings.Contains(arg, "&&") || strings.Contains(arg, ";") || strings.Contains(arg, "$(") {
				t.Errorf("convert container %q args look like a shell command: %q", c.Name, arg)
			}
		}
	}
}

// --- Karpenter Annotations ---

func TestSecurityBatchJobsHaveDoNotDisrupt(t *testing.T) {
	cfg := secConfig()
	sized := secSized()
	ns := "test-ns"

	cj := training.ConvertJobManifest(cfg, sized, ns, pipeline.NewPipelineContext())
	if cj.Spec.Template.Annotations["karpenter.sh/do-not-disrupt"] != "true" {
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
