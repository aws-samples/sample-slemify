// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package build

import (
	"context"
	"fmt"
	"sync"
)

// ContainerDef defines a container to build.
type ContainerDef struct {
	Name       string // e.g., "data-pipeline"
	ContextDir string // e.g., "containers/data-pipeline"
}

// AllContainers returns the list of all buildable containers.
func AllContainers() []ContainerDef {
	return []ContainerDef{
		{Name: "data-pipeline", ContextDir: "containers/data-pipeline"},
	}
}

// BuildResult holds the outcome of a single arch build.
type BuildResult struct {
	Arch     string
	ImageURI string
	Error    error
}

// BuildCommands generates the shell commands to build and push a container on a remote instance.
func BuildCommands(registryURL, containerName, tag, contextDir, arch string, ecrPassword string) []string {
	imageTag := fmt.Sprintf("%s/slemify/%s:%s-%s", registryURL, containerName, tag, arch)
	return []string{
		// ECR login (password is base64-encoded, safe for shell)
		fmt.Sprintf("echo '%s' | docker login --username AWS --password-stdin %s", ecrPassword, registryURL),
		// Build
		fmt.Sprintf("docker build -t %s %s", imageTag, contextDir),
		// Push
		fmt.Sprintf("docker push %s", imageTag),
	}
}

// ManifestCommands generates the shell commands to create and push a multi-arch manifest.
func ManifestCommands(registryURL, containerName, tag string) []string {
	manifestTag := fmt.Sprintf("%s/slemify/%s:%s", registryURL, containerName, tag)
	amd64Tag := fmt.Sprintf("%s-amd64", manifestTag)
	arm64Tag := fmt.Sprintf("%s-arm64", manifestTag)

	return []string{
		fmt.Sprintf("docker manifest create %s --amend %s --amend %s", manifestTag, amd64Tag, arm64Tag),
		fmt.Sprintf("docker manifest push %s", manifestTag),
	}
}

// BuildPlan describes the full build plan for display in dry-run mode.
type BuildPlan struct {
	Containers  []ContainerDef
	Arches      []string
	RegistryURL string
	Tag         string
}

// PrintPlan displays the build plan without executing anything.
func PrintPlan(plan *BuildPlan) {
	fmt.Printf("Build Plan:\n")
	fmt.Printf("  Registry: %s\n", plan.RegistryURL)
	fmt.Printf("  Tag: %s\n", plan.Tag)
	fmt.Printf("  Architectures: %v\n", plan.Arches)
	fmt.Printf("  Containers:\n")
	for _, c := range plan.Containers {
		fmt.Printf("    - %s (%s)\n", c.Name, c.ContextDir)
		for _, arch := range plan.Arches {
			fmt.Printf("      → %s/slemify/%s:%s-%s\n", plan.RegistryURL, c.Name, plan.Tag, arch)
		}
		fmt.Printf("      → %s/slemify/%s:%s (multi-arch)\n", plan.RegistryURL, c.Name, plan.Tag)
	}
	fmt.Printf("\n  EC2 instances:\n")
	for _, spec := range DefaultBuildSpecs() {
		fmt.Printf("    - %s (%s) — launched, used for build, then terminated\n", spec.Arch, spec.InstanceType)
	}
}

// ParallelBuild runs builds on two instances in parallel and collects results.
func ParallelBuild(ctx context.Context, instances []*BuildInstance, buildFn func(ctx context.Context, inst *BuildInstance) error) []error {
	var wg sync.WaitGroup
	errs := make([]error, len(instances))

	for i, inst := range instances {
		wg.Add(1)
		go func(idx int, inst *BuildInstance) {
			defer wg.Done()
			errs[idx] = buildFn(ctx, inst)
		}(i, inst)
	}

	wg.Wait()
	return errs
}
