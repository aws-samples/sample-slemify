// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"context"
	"fmt"

	"github.com/aws-samples/sample-slemify/pkg/build"
	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/spf13/cobra"
)

var buildCmd = &cobra.Command{
	Use:   "build",
	Short: "Build and push multi-arch container images to ECR",
	Long: `Launches EC2 instances (x86 + arm64), builds container images natively on each
via SSM (no SSH keys required), pushes to ECR, creates multi-arch manifests,
and terminates the instances.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		dryRun, _ := cmd.Flags().GetBool("dry-run")
		container, _ := cmd.Flags().GetString("container")
		tag, _ := cmd.Flags().GetString("tag")
		instanceProfile, _ := cmd.Flags().GetString("instance-profile")
		ctx := cmd.Context()

		cfg, _, err := config.Load(cfgFile)
		if err != nil {
			return fmt.Errorf("loading config: %w", err)
		}

		// Determine which containers to build
		containers := build.AllContainers()
		if container != "" {
			filtered := filterContainers(containers, container)
			if len(filtered) == 0 {
				return fmt.Errorf("unknown container %q (valid: data-pipeline)", container)
			}
			containers = filtered
		}

		if dryRun {
			return buildDryRun(ctx, containers, tag)
		}

		return buildAndPush(ctx, containers, tag, instanceProfile, cfg.Data.Bucket)
	},
}

func buildDryRun(ctx context.Context, containers []build.ContainerDef, tag string) error {
	ecrMgr, err := build.NewECRManager(ctx)
	if err != nil {
		return fmt.Errorf("initializing ECR: %w", err)
	}

	plan := &build.BuildPlan{
		Containers:  containers,
		Arches:      []string{"amd64", "arm64"},
		RegistryURL: ecrMgr.RegistryURL(),
		Tag:         tag,
	}
	build.PrintPlan(plan)
	return nil
}

func buildAndPush(ctx context.Context, containers []build.ContainerDef, tag, instanceProfile, s3Bucket string) error {
	ecrMgr, err := build.NewECRManager(ctx)
	if err != nil {
		return fmt.Errorf("initializing ECR: %w", err)
	}

	fmt.Printf("Registry: %s\n", ecrMgr.RegistryURL())
	fmt.Printf("Region: %s\n", ecrMgr.Region())
	fmt.Printf("Access: SSM (no SSH keys)\n\n")

	// Ensure ECR repositories exist
	fmt.Println("Ensuring ECR repositories...")
	uris, err := ecrMgr.EnsureRepositories(ctx)
	if err != nil {
		return fmt.Errorf("creating ECR repositories: %w", err)
	}
	for _, uri := range uris {
		fmt.Printf("  %s\n", uri)
	}
	fmt.Println()

	// Initialize EC2 builder
	ec2Builder, err := build.NewEC2Builder(ctx)
	if err != nil {
		return fmt.Errorf("initializing EC2 builder: %w", err)
	}

	// Ensure instance profile exists for SSM access
	if instanceProfile == "" {
		fmt.Println("Ensuring build instance profile...")
		profile, err := build.EnsureBuildInstanceProfile(ctx)
		if err != nil {
			return fmt.Errorf("setting up build instance profile: %w", err)
		}
		instanceProfile = profile
		fmt.Printf("  Using instance profile: %s\n\n", instanceProfile)
	}

	// Launch build instances
	fmt.Println("Launching build instances...")
	specs := build.DefaultBuildSpecs()
	for i := range specs {
		specs[i].InstanceProfile = instanceProfile
	}

	var instances []*build.BuildInstance
	defer func() {
		fmt.Println("\nTerminating build instances...")
		ec2Builder.TerminateAll(ctx, instances)
	}()

	for _, spec := range specs {
		inst, err := ec2Builder.LaunchBuildInstance(ctx, spec)
		if err != nil {
			return fmt.Errorf("launching %s instance: %w", spec.Arch, err)
		}
		instances = append(instances, inst)
	}

	// Wait for instances to be running
	fmt.Println("\nWaiting for instances...")
	for _, inst := range instances {
		if err := ec2Builder.WaitForReady(ctx, inst, 300_000_000_000); err != nil { // 5 min
			return fmt.Errorf("instance %s not ready: %w", inst.InstanceID, err)
		}
	}

	// Get ECR credentials
	_, ecrPassword, _, err := ecrMgr.GetLoginCredentials(ctx)
	if err != nil {
		return fmt.Errorf("getting ECR credentials: %w", err)
	}

	// Build containers in parallel on both instances
	fmt.Printf("\nBuilding %d containers...\n", len(containers))
	for _, c := range containers {
		fmt.Printf("\n--- %s ---\n", c.Name)

		sourceDir, err := build.ResolveSourceDir(c)
		if err != nil {
			return fmt.Errorf("resolving source for %s: %w", c.Name, err)
		}

		errs := build.ParallelBuild(ctx, instances, func(ctx context.Context, inst *build.BuildInstance) error {
			return build.RemoteBuild(ctx, build.RemoteBuildConfig{
				Instance:    inst,
				S3Bucket:    s3Bucket,
				RegistryURL: ecrMgr.RegistryURL(),
				ECRPassword: ecrPassword,
				Container:   c,
				Tag:         tag,
				SourceDir:   sourceDir,
			})
		})

		for i, err := range errs {
			if err != nil {
				return fmt.Errorf("%s build failed on %s: %w", c.Name, instances[i].Arch, err)
			}
		}

		// Create multi-arch manifest
		fmt.Printf("  Creating multi-arch manifest...\n")
		if err := build.CreateMultiArchManifest(ctx, instances[0],
			ecrMgr.RegistryURL(), ecrPassword, c.Name, tag); err != nil {
			return fmt.Errorf("creating manifest for %s: %w", c.Name, err)
		}
		fmt.Printf("  Done: %s\n", ecrMgr.ImageURI(c.Name, tag))
	}

	fmt.Printf("\nAll containers built and pushed\n")
	fmt.Printf("Registry: %s\n", ecrMgr.RegistryURL())
	fmt.Printf("Deploy will auto-detect this registry on the same AWS account.\n")
	return nil
}

func filterContainers(all []build.ContainerDef, name string) []build.ContainerDef {
	var filtered []build.ContainerDef
	for _, c := range all {
		if c.Name == name {
			filtered = append(filtered, c)
		}
	}
	return filtered
}

func init() {
	buildCmd.Flags().Bool("dry-run", false, "Show build plan without executing")
	buildCmd.Flags().String("container", "", "Build a specific container (default: all)")
	buildCmd.Flags().String("tag", "latest", "Image tag")
	buildCmd.Flags().String("instance-profile", "", "IAM instance profile name for SSM access (must include AmazonSSMManagedInstanceCore)")
	rootCmd.AddCommand(buildCmd)
}
