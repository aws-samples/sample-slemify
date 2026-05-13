// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package build

import (
	"context"
	"encoding/base64"
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/ec2"
	ec2types "github.com/aws/aws-sdk-go-v2/service/ec2/types"
	"github.com/aws/aws-sdk-go-v2/service/iam"
	iamtypes "github.com/aws/aws-sdk-go-v2/service/iam/types"
	"github.com/aws/aws-sdk-go-v2/service/ssm"
)

// BuildInstance represents a remote EC2 instance used for building container images.
type BuildInstance struct {
	InstanceID string
	Arch       string // "amd64" or "arm64"
}

// EC2Builder manages EC2 instances for container builds.
type EC2Builder struct {
	ec2Client *ec2.Client
	ssmClient *ssm.Client
	region    string
}

// NewEC2Builder creates an EC2 builder from the default AWS config.
func NewEC2Builder(ctx context.Context) (*EC2Builder, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("loading AWS config: %w", err)
	}

	return &EC2Builder{
		ec2Client: ec2.NewFromConfig(cfg),
		ssmClient: ssm.NewFromConfig(cfg),
		region:    cfg.Region,
	}, nil
}

// BuildInstanceSpec defines the configuration for a build instance.
type BuildInstanceSpec struct {
	Arch            string // "amd64" or "arm64"
	InstanceType    string
	SubnetID        string // optional, uses default VPC if empty
	SecurityGroup   string // optional, uses default if empty
	InstanceProfile string // IAM instance profile for SSM access
	ProjectName     string // for cost allocation tags
}

// DefaultBuildSpecs returns the default instance specs for both architectures.
func DefaultBuildSpecs() []BuildInstanceSpec {
	return []BuildInstanceSpec{
		{Arch: "amd64", InstanceType: "c5.large"},
		{Arch: "arm64", InstanceType: "c6g.large"},
	}
}

// UserDataScript returns the cloud-init script that installs Docker.
func UserDataScript() string {
	return `#!/bin/bash
set -e
yum install -y docker
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user
touch /tmp/build-ready
`
}

// LaunchBuildInstance launches a single EC2 instance for building.
func (b *EC2Builder) LaunchBuildInstance(ctx context.Context, spec BuildInstanceSpec) (*BuildInstance, error) {
	amiID, err := b.resolveAMI(ctx, spec.Arch)
	if err != nil {
		return nil, fmt.Errorf("resolving AMI for %s: %w", spec.Arch, err)
	}

	input := &ec2.RunInstancesInput{
		ImageId:      aws.String(amiID),
		InstanceType: ec2types.InstanceType(spec.InstanceType),
		MinCount:     aws.Int32(1),
		MaxCount:     aws.Int32(1),
		UserData:     aws.String(EncodeUserData(UserDataScript())),
		TagSpecifications: []ec2types.TagSpecification{
			{
				ResourceType: ec2types.ResourceTypeInstance,
				Tags: []ec2types.Tag{
					{Key: aws.String("Name"), Value: aws.String(fmt.Sprintf("slemify-build-%s", spec.Arch))},
					{Key: aws.String("slemify.io/purpose"), Value: aws.String("container-build")},
					{Key: aws.String("slemify.io/arch"), Value: aws.String(spec.Arch)},
					{Key: aws.String("slemify.io/project"), Value: aws.String(spec.ProjectName)},
				},
			},
		},
	}

	if spec.InstanceProfile != "" {
		input.IamInstanceProfile = &ec2types.IamInstanceProfileSpecification{
			Name: aws.String(spec.InstanceProfile),
		}
	}
	if spec.SecurityGroup != "" {
		input.SecurityGroupIds = []string{spec.SecurityGroup}
	}
	if spec.SubnetID != "" {
		input.SubnetId = aws.String(spec.SubnetID)
	}

	result, err := b.ec2Client.RunInstances(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("launching %s instance: %w", spec.Arch, err)
	}

	if len(result.Instances) == 0 {
		return nil, fmt.Errorf("no instances returned for %s", spec.Arch)
	}

	instanceID := aws.ToString(result.Instances[0].InstanceId)
	fmt.Printf("  Launched %s build instance: %s\n", spec.Arch, instanceID)

	return &BuildInstance{
		InstanceID: instanceID,
		Arch:       spec.Arch,
	}, nil
}

// WaitForReady waits until the instance is running.
func (b *EC2Builder) WaitForReady(ctx context.Context, instance *BuildInstance, timeout time.Duration) error {
	waiter := ec2.NewInstanceRunningWaiter(b.ec2Client)
	err := waiter.Wait(ctx, &ec2.DescribeInstancesInput{
		InstanceIds: []string{instance.InstanceID},
	}, timeout)
	if err != nil {
		return fmt.Errorf("waiting for instance %s: %w", instance.InstanceID, err)
	}

	fmt.Printf("  %s instance running: %s\n", instance.Arch, instance.InstanceID)
	return nil
}

// TerminateInstance terminates a build instance.
func (b *EC2Builder) TerminateInstance(ctx context.Context, instanceID string) error {
	_, err := b.ec2Client.TerminateInstances(ctx, &ec2.TerminateInstancesInput{
		InstanceIds: []string{instanceID},
	})
	if err != nil {
		return fmt.Errorf("terminating instance %s: %w", instanceID, err)
	}
	fmt.Printf("  Terminated instance: %s\n", instanceID)
	return nil
}

// TerminateAll terminates a list of build instances. Logs errors but doesn't fail.
func (b *EC2Builder) TerminateAll(ctx context.Context, instances []*BuildInstance) {
	for _, inst := range instances {
		if inst != nil && inst.InstanceID != "" {
			if err := b.TerminateInstance(ctx, inst.InstanceID); err != nil {
				fmt.Printf("  Warning: failed to terminate %s: %v\n", inst.InstanceID, err)
			}
		}
	}
}

// resolveAMI finds the latest Amazon Linux 2023 AMI for the given architecture.
func (b *EC2Builder) resolveAMI(ctx context.Context, arch string) (string, error) {
	ssmArch := "x86_64"
	if arch == "arm64" {
		ssmArch = "arm64"
	}

	paramName := fmt.Sprintf("/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-%s", ssmArch)
	result, err := b.ssmClient.GetParameter(ctx, &ssm.GetParameterInput{
		Name: aws.String(paramName),
	})
	if err != nil {
		return "", fmt.Errorf("resolving AMI via SSM parameter %s: %w", paramName, err)
	}

	return aws.ToString(result.Parameter.Value), nil
}

// EncodeUserData base64-encodes the user-data script for EC2.
func EncodeUserData(script string) string {
	return base64.StdEncoding.EncodeToString([]byte(script))
}

const buildProfileName = "slemify-build"
const buildRoleName = "slemify-build"

const assumeRolePolicy = `{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}`

// EnsureBuildInstanceProfile returns an instance profile with SSM, ECR,
// and S3 access for build instances. Creates the role and profile if they
// don't already exist.
func EnsureBuildInstanceProfile(ctx context.Context) (string, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return "", fmt.Errorf("loading AWS config: %w", err)
	}
	iamClient := iam.NewFromConfig(cfg)

	// Check if instance profile already exists
	_, err = iamClient.GetInstanceProfile(ctx, &iam.GetInstanceProfileInput{
		InstanceProfileName: aws.String(buildProfileName),
	})
	if err == nil {
		return buildProfileName, nil
	}

	// Create the IAM role
	fmt.Printf("  Creating IAM role %s...\n", buildRoleName)
	_, err = iamClient.CreateRole(ctx, &iam.CreateRoleInput{
		RoleName:                 aws.String(buildRoleName),
		AssumeRolePolicyDocument: aws.String(assumeRolePolicy),
		Description:              aws.String("Slemify build instances - SSM, ECR, and S3 access for container image builds"),
		Tags: []iamtypes.Tag{
			{Key: aws.String("app.kubernetes.io/managed-by"), Value: aws.String("slemify")},
		},
	})
	if err != nil {
		return "", fmt.Errorf("creating role: %w", err)
	}

	// Attach managed policies
	policies := []string{
		"arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
		"arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser",
		"arn:aws:iam::aws:policy/AmazonS3FullAccess",
	}
	for _, arn := range policies {
		_, err = iamClient.AttachRolePolicy(ctx, &iam.AttachRolePolicyInput{
			RoleName:  aws.String(buildRoleName),
			PolicyArn: aws.String(arn),
		})
		if err != nil {
			return "", fmt.Errorf("attaching policy %s: %w", arn, err)
		}
	}

	// Create instance profile and add role
	fmt.Printf("  Creating instance profile %s...\n", buildProfileName)
	_, err = iamClient.CreateInstanceProfile(ctx, &iam.CreateInstanceProfileInput{
		InstanceProfileName: aws.String(buildProfileName),
		Tags: []iamtypes.Tag{
			{Key: aws.String("app.kubernetes.io/managed-by"), Value: aws.String("slemify")},
		},
	})
	if err != nil {
		return "", fmt.Errorf("creating instance profile: %w", err)
	}

	_, err = iamClient.AddRoleToInstanceProfile(ctx, &iam.AddRoleToInstanceProfileInput{
		InstanceProfileName: aws.String(buildProfileName),
		RoleName:            aws.String(buildRoleName),
	})
	if err != nil {
		return "", fmt.Errorf("adding role to instance profile: %w", err)
	}

	// IAM propagation delay — profile must be visible to EC2 before launch
	fmt.Printf("  Waiting for IAM propagation...\n")
	time.Sleep(10 * time.Second)

	return buildProfileName, nil
}
