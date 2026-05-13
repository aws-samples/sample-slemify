// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package build

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// RemoteBuildConfig holds all parameters for a remote container build.
type RemoteBuildConfig struct {
	Instance    *BuildInstance
	S3Bucket    string // S3 bucket for transferring build context
	RegistryURL string
	ECRPassword string
	Container   ContainerDef
	Tag         string
	SourceDir   string // local path to the container source
}

// RemoteBuild connects to an EC2 instance via SSM, uploads the container source
// through S3, builds the Docker image, and pushes it to ECR.
func RemoteBuild(ctx context.Context, cfg RemoteBuildConfig) error {
	// Wait for SSM agent to be ready
	fmt.Printf("  [%s] Waiting for SSM agent on %s...\n", cfg.Instance.Arch, cfg.Instance.InstanceID)
	if err := WaitForSSMReady(ctx, cfg.Instance.InstanceID, 5*time.Minute); err != nil {
		return fmt.Errorf("SSM not ready: %w", err)
	}

	// Connect via SSM
	client, err := NewSSMClient(ctx, cfg.Instance.InstanceID)
	if err != nil {
		return fmt.Errorf("SSM connect: %w", err)
	}

	// Wait for Docker to be ready (user-data may still be running)
	fmt.Printf("  [%s] Waiting for Docker...\n", cfg.Instance.Arch)
	for i := 0; i < 60; i++ {
		if _, err := client.Run(ctx, "sudo docker info"); err == nil {
			break
		}
		if i == 59 {
			return fmt.Errorf("Docker not ready after 300s")
		}
		time.Sleep(5 * time.Second)
	}

	// Upload container source via S3
	fmt.Printf("  [%s] Uploading %s source via S3...\n", cfg.Instance.Arch, cfg.Container.Name)
	s3Key := fmt.Sprintf("build-context/%s/%s.tar", cfg.Container.Name, cfg.Instance.Arch)
	if err := uploadBuildContext(ctx, cfg.S3Bucket, s3Key, cfg.SourceDir); err != nil {
		return fmt.Errorf("uploading build context to S3: %w", err)
	}

	// Download build context on the instance
	remoteDir := fmt.Sprintf("/home/ec2-user/build/%s", cfg.Container.Name)
	downloadCmd := fmt.Sprintf(
		"mkdir -p %s && aws s3 cp s3://%s/%s - | tar xf - -C %s",
		remoteDir, cfg.S3Bucket, s3Key, remoteDir)
	if _, err := client.Run(ctx, downloadCmd); err != nil {
		return fmt.Errorf("downloading build context: %w", err)
	}

	// ECR login
	fmt.Printf("  [%s] Logging into ECR...\n", cfg.Instance.Arch)
	loginCmd := fmt.Sprintf("echo '%s' | sudo docker login --username AWS --password-stdin %s",
		cfg.ECRPassword, cfg.RegistryURL)
	if _, err := client.Run(ctx, loginCmd); err != nil {
		return fmt.Errorf("ECR login: %w", err)
	}

	// Build
	imageTag := fmt.Sprintf("%s/slemify/%s:%s-%s",
		cfg.RegistryURL, cfg.Container.Name, cfg.Tag, cfg.Instance.Arch)
	fmt.Printf("  [%s] Building %s...\n", cfg.Instance.Arch, imageTag)
	buildCmd := fmt.Sprintf("sudo docker build -t %s %s", imageTag, remoteDir)
	if output, err := client.Run(ctx, buildCmd); err != nil {
		return fmt.Errorf("docker build: %w\n%s", err, lastLines(output, 20))
	}

	// Push
	fmt.Printf("  [%s] Pushing %s...\n", cfg.Instance.Arch, imageTag)
	pushCmd := fmt.Sprintf("sudo docker push %s", imageTag)
	if output, err := client.Run(ctx, pushCmd); err != nil {
		return fmt.Errorf("docker push: %w\n%s", err, lastLines(output, 10))
	}

	fmt.Printf("  [%s] Done: %s\n", cfg.Instance.Arch, imageTag)
	return nil
}

// CreateMultiArchManifest creates and pushes a multi-arch manifest via SSM.
func CreateMultiArchManifest(ctx context.Context, instance *BuildInstance,
	registryURL, ecrPassword, containerName, tag string) error {

	client, err := NewSSMClient(ctx, instance.InstanceID)
	if err != nil {
		return fmt.Errorf("SSM connect for manifest: %w", err)
	}

	loginCmd := fmt.Sprintf("echo '%s' | sudo docker login --username AWS --password-stdin %s",
		ecrPassword, registryURL)
	if _, err := client.Run(ctx, loginCmd); err != nil {
		return fmt.Errorf("ECR login for manifest: %w", err)
	}

	for _, arch := range []string{"amd64", "arm64"} {
		pullTag := fmt.Sprintf("%s/slemify/%s:%s-%s", registryURL, containerName, tag, arch)
		if _, err := client.Run(ctx, fmt.Sprintf("sudo docker pull %s", pullTag)); err != nil {
			return fmt.Errorf("pulling %s: %w", pullTag, err)
		}
	}

	cmds := ManifestCommands(registryURL, containerName, tag)
	for _, cmd := range cmds {
		if _, err := client.Run(ctx, "sudo "+cmd); err != nil {
			return fmt.Errorf("manifest command failed: %w", err)
		}
	}

	return nil
}

// uploadBuildContext creates a tar of the source directory and uploads it to S3.
func uploadBuildContext(ctx context.Context, bucket, key, sourceDir string) error {
	var buf bytes.Buffer
	if err := tarDirectory(sourceDir, &buf); err != nil {
		return fmt.Errorf("creating tar: %w", err)
	}

	awsCfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return fmt.Errorf("loading AWS config: %w", err)
	}

	s3Client := s3.NewFromConfig(awsCfg)
	_, err = s3Client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
		Body:   bytes.NewReader(buf.Bytes()),
	})
	if err != nil {
		return fmt.Errorf("uploading to s3://%s/%s: %w", bucket, key, err)
	}

	return nil
}

// lastLines returns the last n lines of a string.
func lastLines(s string, n int) string {
	lines := splitLines(s)
	if len(lines) <= n {
		return s
	}
	result := ""
	for _, line := range lines[len(lines)-n:] {
		result += line + "\n"
	}
	return result
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

// ResolveSourceDir finds the local path to a container's source directory.
func ResolveSourceDir(containerDef ContainerDef) (string, error) {
	if info, err := os.Stat(containerDef.ContextDir); err == nil && info.IsDir() {
		return containerDef.ContextDir, nil
	}

	srcPath := fmt.Sprintf("src/%s", containerDef.ContextDir)
	if info, err := os.Stat(srcPath); err == nil && info.IsDir() {
		return srcPath, nil
	}

	return "", fmt.Errorf("container source directory not found: %s", containerDef.ContextDir)
}
