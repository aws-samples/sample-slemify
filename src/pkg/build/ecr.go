// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Package build handles container image building on remote EC2 instances
// and ECR repository management.
package build

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/ecr"
	ecrtypes "github.com/aws/aws-sdk-go-v2/service/ecr/types"
	"github.com/aws/aws-sdk-go-v2/service/sts"
)

// ECRManager handles ECR repository creation and authentication.
type ECRManager struct {
	ecrClient *ecr.Client
	stsClient *sts.Client
	region    string
	accountID string
}

// NewECRManager creates an ECR manager from the default AWS config.
func NewECRManager(ctx context.Context) (*ECRManager, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("loading AWS config: %w", err)
	}

	stsClient := sts.NewFromConfig(cfg)
	identity, err := stsClient.GetCallerIdentity(ctx, &sts.GetCallerIdentityInput{})
	if err != nil {
		return nil, fmt.Errorf("getting caller identity: %w", err)
	}

	return &ECRManager{
		ecrClient: ecr.NewFromConfig(cfg),
		stsClient: stsClient,
		region:    cfg.Region,
		accountID: aws.ToString(identity.Account),
	}, nil
}

// RegistryURL returns the ECR registry URL for this account and region.
func (m *ECRManager) RegistryURL() string {
	return fmt.Sprintf("%s.dkr.ecr.%s.amazonaws.com", m.accountID, m.region)
}

// Region returns the AWS region.
func (m *ECRManager) Region() string {
	return m.region
}

// AccountID returns the AWS account ID.
func (m *ECRManager) AccountID() string {
	return m.accountID
}

// ContainerRepos returns the list of ECR repository names slemify needs,
// derived from the buildable containers so the two never drift.
func ContainerRepos() []string {
	var repos []string
	for _, c := range AllContainers() {
		repos = append(repos, "slemify/"+c.Name)
	}
	return repos
}

// EnsureRepositories creates ECR repositories if they don't exist.
// Returns the list of repository URIs.
func (m *ECRManager) EnsureRepositories(ctx context.Context) ([]string, error) {
	var uris []string

	for _, repoName := range ContainerRepos() {
		uri, err := m.ensureRepo(ctx, repoName)
		if err != nil {
			return nil, fmt.Errorf("ensuring repo %s: %w", repoName, err)
		}
		uris = append(uris, uri)
	}

	return uris, nil
}

// EnsureRepository creates a single ECR repository if it doesn't exist.
func (m *ECRManager) EnsureRepository(ctx context.Context, repoName string) (string, error) {
	return m.ensureRepo(ctx, repoName)
}

func (m *ECRManager) ensureRepo(ctx context.Context, repoName string) (string, error) {
	// Check if it exists
	desc, err := m.ecrClient.DescribeRepositories(ctx, &ecr.DescribeRepositoriesInput{
		RepositoryNames: []string{repoName},
	})
	if err == nil && len(desc.Repositories) > 0 {
		return aws.ToString(desc.Repositories[0].RepositoryUri), nil
	}

	// Create it
	result, err := m.ecrClient.CreateRepository(ctx, &ecr.CreateRepositoryInput{
		RepositoryName:     aws.String(repoName),
		ImageTagMutability: "MUTABLE",
		Tags: []ecrtypes.Tag{
			{Key: aws.String("app.kubernetes.io/managed-by"), Value: aws.String("slemify")},
			{Key: aws.String("slemify.io/purpose"), Value: aws.String("container-images")},
		},
	})
	if err != nil {
		// Handle race condition: repo created between describe and create
		if strings.Contains(err.Error(), "RepositoryAlreadyExistsException") {
			return fmt.Sprintf("%s/%s", m.RegistryURL(), repoName), nil
		}
		return "", fmt.Errorf("creating repository %s: %w", repoName, err)
	}

	return aws.ToString(result.Repository.RepositoryUri), nil
}

// GetLoginCredentials returns the Docker login password for ECR.
func (m *ECRManager) GetLoginCredentials(ctx context.Context) (username, password, endpoint string, err error) {
	result, err := m.ecrClient.GetAuthorizationToken(ctx, &ecr.GetAuthorizationTokenInput{})
	if err != nil {
		return "", "", "", fmt.Errorf("getting ECR auth token: %w", err)
	}

	if len(result.AuthorizationData) == 0 {
		return "", "", "", fmt.Errorf("no authorization data returned")
	}

	auth := result.AuthorizationData[0]
	decoded, err := base64.StdEncoding.DecodeString(aws.ToString(auth.AuthorizationToken))
	if err != nil {
		return "", "", "", fmt.Errorf("decoding auth token: %w", err)
	}

	parts := strings.SplitN(string(decoded), ":", 2)
	if len(parts) != 2 {
		return "", "", "", fmt.Errorf("unexpected auth token format")
	}

	return parts[0], parts[1], aws.ToString(auth.ProxyEndpoint), nil
}

// ImageURI returns the full image URI for a container name and tag.
func (m *ECRManager) ImageURI(containerName, tag string) string {
	return fmt.Sprintf("%s/slemify/%s:%s", m.RegistryURL(), containerName, tag)
}
