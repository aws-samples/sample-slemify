// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package build

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

func TestAllContainers(t *testing.T) {
	containers := AllContainers()

	if len(containers) != 4 {
		t.Fatalf("AllContainers() returned %d, want 4", len(containers))
	}

	names := map[string]bool{}
	for _, c := range containers {
		names[c.Name] = true
		if c.ContextDir == "" {
			t.Errorf("container %s has empty context dir", c.Name)
		}
	}

	for _, expected := range []string{"data-pipeline", "classifier-trainer", "classifier-serving", "gguf-convert"} {
		if !names[expected] {
			t.Errorf("missing container %q", expected)
		}
	}
}

func TestBuildCommands(t *testing.T) {
	cmds := BuildCommands(
		"123456789012.dkr.ecr.us-east-1.amazonaws.com",
		"data-pipeline", "latest",
		"containers/data-pipeline", "amd64",
		"test-password",
	)

	if len(cmds) != 3 {
		t.Fatalf("BuildCommands returned %d commands, want 3", len(cmds))
	}

	// Login command
	if !strings.Contains(cmds[0], "docker login") {
		t.Error("first command should be docker login")
	}
	if !strings.Contains(cmds[0], "123456789012.dkr.ecr.us-east-1.amazonaws.com") {
		t.Error("login should target ECR registry")
	}

	// Build command
	if !strings.Contains(cmds[1], "docker build") {
		t.Error("second command should be docker build")
	}
	if !strings.Contains(cmds[1], "latest-amd64") {
		t.Error("build tag should include arch suffix")
	}
	if !strings.Contains(cmds[1], "containers/data-pipeline") {
		t.Error("build should reference context dir")
	}

	// Push command
	if !strings.Contains(cmds[2], "docker push") {
		t.Error("third command should be docker push")
	}
}

func TestManifestCommands(t *testing.T) {
	cmds := ManifestCommands(
		"123456789012.dkr.ecr.us-east-1.amazonaws.com",
		"data-pipeline", "latest",
	)

	if len(cmds) != 2 {
		t.Fatalf("ManifestCommands returned %d commands, want 2", len(cmds))
	}

	// Create command
	if !strings.Contains(cmds[0], "docker manifest create") {
		t.Error("first command should create manifest")
	}
	if !strings.Contains(cmds[0], "latest-amd64") {
		t.Error("manifest should reference amd64 tag")
	}
	if !strings.Contains(cmds[0], "latest-arm64") {
		t.Error("manifest should reference arm64 tag")
	}

	// Push command
	if !strings.Contains(cmds[1], "docker manifest push") {
		t.Error("second command should push manifest")
	}
	if !strings.Contains(cmds[1], "data-pipeline:latest") {
		t.Error("pushed manifest should use clean tag")
	}
}

func TestParallelBuild(t *testing.T) {
	instances := []*BuildInstance{
		{InstanceID: "i-amd64", Arch: "amd64"},
		{InstanceID: "i-arm64", Arch: "arm64"},
	}

	executed := make([]string, 2)
	errs := ParallelBuild(context.Background(), instances, func(ctx context.Context, inst *BuildInstance) error {
		if inst.Arch == "amd64" {
			executed[0] = "amd64"
		} else {
			executed[1] = "arm64"
		}
		return nil
	})

	for _, err := range errs {
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
	if executed[0] != "amd64" || executed[1] != "arm64" {
		t.Errorf("not all arches executed: %v", executed)
	}
}

func TestParallelBuildWithError(t *testing.T) {
	instances := []*BuildInstance{
		{InstanceID: "i-amd64", Arch: "amd64"},
		{InstanceID: "i-arm64", Arch: "arm64"},
	}

	errs := ParallelBuild(context.Background(), instances, func(ctx context.Context, inst *BuildInstance) error {
		if inst.Arch == "arm64" {
			return fmt.Errorf("arm64 build failed")
		}
		return nil
	})

	if errs[0] != nil {
		t.Error("amd64 should succeed")
	}
	if errs[1] == nil {
		t.Error("arm64 should fail")
	}
}

func TestPrintPlan(t *testing.T) {
	plan := &BuildPlan{
		Containers:  AllContainers(),
		Arches:      []string{"amd64", "arm64"},
		RegistryURL: "123456789012.dkr.ecr.us-east-1.amazonaws.com",
		Tag:         "latest",
	}
	PrintPlan(plan) // should not panic — 2 containers × 2 arches
}
