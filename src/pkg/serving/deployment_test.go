// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"strings"
	"testing"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func karpenterConfig() *config.ExpertConfig {
	return &config.ExpertConfig{
		APIVersion: "slemify/v1",
		Project: config.ProjectConfig{
			Name:          "karpenter-expert",
			Domain:        "Karpenter configuration and optimization on EKS",
			DomainVersion: "1.2",
		},
		Model: config.ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		Data: config.DataConfig{
			Bucket: "my-bucket",
			Path:   "karpenter-data/",
			Synthetic: config.SyntheticConfig{Model: "claude-sonnet", Pairs: 500},
		},
		Training: config.TrainingConfig{Spot: true},
	}
}

func sized7B() config.SizedConfig {
	return config.AutoSize(
		config.ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		config.DataConfig{Bucket: "b", Path: "p/", Synthetic: config.SyntheticConfig{Model: "m", Pairs: 500}},
		config.TrainingConfig{},
	)
}

var pc = pipeline.NewPipelineContext()

// --- Deployment Tests ---

func TestDeploymentLabels(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	dep := m.Deployment
	if dep.Name != "karpenter-expert-inference" {
		t.Errorf("name = %q, want %q", dep.Name, "karpenter-expert-inference")
	}
	if dep.Namespace != "slemify" {
		t.Errorf("namespace = %q, want %q", dep.Namespace, "slemify")
	}
	if dep.Labels["slemify.io/project"] != "karpenter-expert" {
		t.Errorf("project label = %q", dep.Labels["slemify.io/project"])
	}
	if dep.Labels["slemify.io/stage"] != "serving" {
		t.Errorf("stage label = %q", dep.Labels["slemify.io/stage"])
	}
	if dep.Labels["app.kubernetes.io/managed-by"] != "slemify" {
		t.Errorf("managed-by label = %q", dep.Labels["app.kubernetes.io/managed-by"])
	}
}

func TestDeploymentArm64NodeSelector(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	ns := m.Deployment.Spec.Template.Spec.NodeSelector
	if ns["kubernetes.io/arch"] != "" {
		t.Errorf("arch should not be constrained, got %q — let Karpenter pick", ns["kubernetes.io/arch"])
	}
	if ns["slemify.io/workload"] != "slm" {
		t.Errorf("workload = %q, want slm", ns["slemify.io/workload"])
	}
}

func TestDeploymentInitContainer(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	initContainers := m.Deployment.Spec.Template.Spec.InitContainers
	if len(initContainers) != 1 {
		t.Fatalf("init containers = %d, want 1", len(initContainers))
	}
	ic := initContainers[0]
	if ic.Name != "model-loader" {
		t.Errorf("init container name = %q, want model-loader", ic.Name)
	}
	if len(ic.VolumeMounts) == 0 {
		t.Error("init container should mount model-storage volume")
	}
}

func TestDeploymentReadinessProbe(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	c := m.Deployment.Spec.Template.Spec.Containers[0]
	if c.ReadinessProbe == nil {
		t.Fatal("readiness probe should be set")
	}
	if c.ReadinessProbe.HTTPGet == nil {
		t.Fatal("readiness probe should use HTTPGet")
	}
	if c.ReadinessProbe.HTTPGet.Path != "/health" {
		t.Errorf("readiness path = %q, want /health", c.ReadinessProbe.HTTPGet.Path)
	}
	if c.ReadinessProbe.HTTPGet.Port != intstr.FromInt32(8080) {
		t.Errorf("readiness port = %v, want 8080", c.ReadinessProbe.HTTPGet.Port)
	}
}

func TestDeploymentLivenessProbe(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	c := m.Deployment.Spec.Template.Spec.Containers[0]
	if c.LivenessProbe == nil {
		t.Fatal("liveness probe should be set")
	}
	if c.LivenessProbe.HTTPGet.Path != "/health" {
		t.Errorf("liveness path = %q, want /health", c.LivenessProbe.HTTPGet.Path)
	}
}

func TestDeploymentResourceRequests(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	c := m.Deployment.Spec.Template.Spec.Containers[0]
	cpu := c.Resources.Requests["cpu"]
	mem := c.Resources.Requests["memory"]
	if cpu.IsZero() {
		t.Error("CPU request should be set")
	}
	if mem.IsZero() {
		t.Error("memory request should be set")
	}
}

func TestDeploymentModelVolume(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	volumes := m.Deployment.Spec.Template.Spec.Volumes
	found := false
	for _, v := range volumes {
		if v.Name == "model-storage" && v.EmptyDir != nil {
			found = true
		}
	}
	if !found {
		t.Error("should have model-storage emptyDir volume")
	}
}

// --- Service Tests ---

func TestServiceSpec(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	svc := m.Service
	if svc.Name != "karpenter-expert-inference" {
		t.Errorf("service name = %q", svc.Name)
	}
	if svc.Spec.Type != "ClusterIP" {
		t.Errorf("service type = %q, want ClusterIP", svc.Spec.Type)
	}
	if len(svc.Spec.Ports) != 1 {
		t.Fatalf("ports = %d, want 1", len(svc.Spec.Ports))
	}
	if svc.Spec.Ports[0].Port != 8080 {
		t.Errorf("port = %d, want 8080", svc.Spec.Ports[0].Port)
	}
	if svc.Spec.Selector["slemify.io/project"] != "karpenter-expert" {
		t.Error("service selector should match deployment labels")
	}
}

// --- PDB Tests ---

func TestPDBSpec(t *testing.T) {
	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	pdb := m.PodDisruptionBudget
	if pdb.Name != "karpenter-expert-inference" {
		t.Errorf("PDB name = %q", pdb.Name)
	}
	if pdb.Spec.MinAvailable == nil {
		t.Fatal("PDB minAvailable should be set")
	}
	if pdb.Spec.MinAvailable.IntValue() != 1 {
		t.Errorf("PDB minAvailable = %d, want 1", pdb.Spec.MinAvailable.IntValue())
	}
	if pdb.Spec.Selector.MatchLabels["slemify.io/project"] != "karpenter-expert" {
		t.Error("PDB selector should match deployment labels")
	}
}

// --- Helper Tests ---

// --- S3 Mount Tests ---

func TestDeploymentS3MountNoInitContainer(t *testing.T) {
	// Enable S3 mount mode
	pc.UseS3Mount = true
	defer func() { pc.UseS3Mount = false }()

	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	initContainers := m.Deployment.Spec.Template.Spec.InitContainers
	if len(initContainers) != 0 {
		t.Errorf("S3 mount mode should have 0 init containers, got %d", len(initContainers))
	}
}

func TestDeploymentS3MountPVCVolume(t *testing.T) {
	pc.UseS3Mount = true
	defer func() { pc.UseS3Mount = false }()

	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	volumes := m.Deployment.Spec.Template.Spec.Volumes
	found := false
	for _, v := range volumes {
		if v.Name == "model-storage" && v.PersistentVolumeClaim != nil {
			if v.PersistentVolumeClaim.ClaimName == S3ModelVolumeName(cfg.Project.Name) {
				found = true
			}
		}
	}
	if !found {
		t.Error("S3 mount mode should have model-storage PVC volume")
	}
}

func TestDeploymentS3MountModelPath(t *testing.T) {
	pc.UseS3Mount = true
	defer func() { pc.UseS3Mount = false }()

	cfg := karpenterConfig()
	sized := sized7B()
	m := GenerateInferenceManifests(cfg, sized, "slemify", pc)

	c := m.Deployment.Spec.Template.Spec.Containers[0]
	// The model path should reference the GGUF filename directly (not model.gguf)
	foundModelArg := false
	for i, arg := range c.Args {
		if arg == "--model" && i+1 < len(c.Args) {
			modelPath := c.Args[i+1]
			expected := "/models/" + cfg.Model.GGUFFilename()
			if modelPath != expected {
				t.Errorf("model path = %q, want %q", modelPath, expected)
			}
			foundModelArg = true
			break
		}
	}
	if !foundModelArg {
		t.Error("--model argument not found in container args")
	}
}

func TestS3MountManifestsContent(t *testing.T) {
	yaml := S3MountManifests("test-project", "test-bucket", "slemify")

	// Should contain PV and PVC
	if !strings.Contains(yaml, "kind: PersistentVolume") {
		t.Error("should contain PersistentVolume")
	}
	if !strings.Contains(yaml, "kind: PersistentVolumeClaim") {
		t.Error("should contain PersistentVolumeClaim")
	}
	if !strings.Contains(yaml, "s3.csi.aws.com") {
		t.Error("should reference S3 CSI driver")
	}
	if !strings.Contains(yaml, "bucketName: test-bucket") {
		t.Error("should reference the bucket name")
	}
	if !strings.Contains(yaml, "read-only") {
		t.Error("should be read-only")
	}
	if !strings.Contains(yaml, "authenticationSource: pod") {
		t.Error("should use pod-level authentication")
	}
	if !strings.Contains(yaml, "prefix models/test-project/") {
		t.Error("should scope to the project's model prefix")
	}
}
