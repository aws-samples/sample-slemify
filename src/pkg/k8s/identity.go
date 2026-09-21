// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package k8s

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/eks"
	"github.com/aws/aws-sdk-go-v2/service/iam"
	iamtypes "github.com/aws/aws-sdk-go-v2/service/iam/types"
	"github.com/aws/aws-sdk-go-v2/service/sts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/clientcmd"
)

// PodIdentityConfig holds the result of Pod Identity setup.
type PodIdentityConfig struct {
	RoleName           string
	RoleARN            string
	ServiceAccountName string
}

// EnsurePodIdentity sets up EKS Pod Identity for the slemify namespace:
// 1. Creates an IAM role with S3 access to the data bucket
// 2. Creates a K8s ServiceAccount
// 3. Creates an EKS Pod Identity association
func (c *Client) EnsurePodIdentity(ctx context.Context, clusterName, bucket, projectName string) (*PodIdentityConfig, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("loading AWS config: %w", err)
	}

	stsClient := sts.NewFromConfig(cfg)
	identity, err := stsClient.GetCallerIdentity(ctx, &sts.GetCallerIdentityInput{})
	if err != nil {
		return nil, fmt.Errorf("getting caller identity: %w", err)
	}
	_ = aws.ToString(identity.Account) // validate we have credentials

	iamClient := iam.NewFromConfig(cfg)
	eksClient := eks.NewFromConfig(cfg)

	roleName := fmt.Sprintf("slemify-%s-data", projectName)
	saName := fmt.Sprintf("slemify-%s", projectName)

	// 1. Create IAM role with Pod Identity trust policy
	roleARN, err := ensureIAMRole(ctx, iamClient, roleName, bucket)
	if err != nil {
		return nil, fmt.Errorf("ensuring IAM role: %w", err)
	}
	fmt.Printf("  IAM role: %s\n", roleARN)

	// 2. Create K8s ServiceAccount
	if err := c.ensureServiceAccount(ctx, saName); err != nil {
		return nil, fmt.Errorf("ensuring service account: %w", err)
	}
	fmt.Printf("  ServiceAccount: %s/%s\n", c.namespace, saName)

	// 3. Create Pod Identity association
	if err := ensurePodIdentityAssociation(ctx, eksClient, clusterName, c.namespace, saName, roleARN); err != nil {
		return nil, fmt.Errorf("ensuring pod identity association: %w", err)
	}
	fmt.Printf("  Pod Identity: associated\n")

	return &PodIdentityConfig{
		RoleName:           roleName,
		RoleARN:            roleARN,
		ServiceAccountName: saName,
	}, nil
}

func ensureIAMRole(ctx context.Context, client *iam.Client, roleName, bucket string) (string, error) {
	// Check if role exists
	getResult, err := client.GetRole(ctx, &iam.GetRoleInput{
		RoleName: aws.String(roleName),
	})
	if err == nil {
		return aws.ToString(getResult.Role.Arn), nil
	}

	// Trust policy for EKS Pod Identity
	trustPolicy := map[string]interface{}{
		"Version": "2012-10-17",
		"Statement": []map[string]interface{}{
			{
				"Effect": "Allow",
				"Principal": map[string]interface{}{
					"Service": "pods.eks.amazonaws.com",
				},
				"Action": []string{
					"sts:AssumeRole",
					"sts:TagSession",
				},
			},
		},
	}
	trustPolicyJSON, _ := json.Marshal(trustPolicy)

	createResult, err := client.CreateRole(ctx, &iam.CreateRoleInput{
		RoleName:                 aws.String(roleName),
		AssumeRolePolicyDocument: aws.String(string(trustPolicyJSON)),
		Description:              aws.String(fmt.Sprintf("Slemify data pipeline role for S3 access to %s", bucket)),
		Tags: []iamtypes.Tag{
			{Key: aws.String("slemify.io/purpose"), Value: aws.String("data-pipeline")},
		},
	})
	if err != nil {
		if strings.Contains(err.Error(), "EntityAlreadyExists") {
			getResult, _ := client.GetRole(ctx, &iam.GetRoleInput{RoleName: aws.String(roleName)})
			return aws.ToString(getResult.Role.Arn), nil
		}
		return "", fmt.Errorf("creating role: %w", err)
	}

	// Attach inline S3 policy — least privilege for the specific bucket
	s3Policy := map[string]interface{}{
		"Version": "2012-10-17",
		"Statement": []map[string]interface{}{
			{
				"Effect": "Allow",
				"Action": []string{
					"s3:GetObject",
					"s3:PutObject",
					"s3:ListBucket",
					"s3:DeleteObject",
					"s3:GetEncryptionConfiguration",
					"s3:PutEncryptionConfiguration",
				},
				"Resource": []string{
					fmt.Sprintf("arn:aws:s3:::%s", bucket),
					fmt.Sprintf("arn:aws:s3:::%s/*", bucket),
				},
			},
			{
				"Effect": "Allow",
				"Action": []string{
					"bedrock:InvokeModel",
					"bedrock:InvokeModelWithResponseStream",
				},
				"Resource": []string{"*"},
			},
			{
				// Marketplace-listed models (Anthropic among them) are only
				// invocable once the account is subscribed to the model, and
				// Bedrock completes that subscription on the calling principal's
				// behalf. Without these two actions the first call succeeds and
				// every later one is denied with "not authorized to perform the
				// required AWS Marketplace actions".
				"Effect": "Allow",
				"Action": []string{
					"aws-marketplace:ViewSubscriptions",
					"aws-marketplace:Subscribe",
				},
				"Resource": []string{"*"},
			},
			{
				"Effect": "Allow",
				"Action": []string{
					"servicequotas:ListServiceQuotas",
				},
				"Resource": []string{"*"},
			},
		},
	}
	s3PolicyJSON, _ := json.Marshal(s3Policy)

	_, err = client.PutRolePolicy(ctx, &iam.PutRolePolicyInput{
		RoleName:       aws.String(roleName),
		PolicyName:     aws.String("slemify-data-pipeline"),
		PolicyDocument: aws.String(string(s3PolicyJSON)),
	})
	if err != nil {
		return "", fmt.Errorf("attaching S3 policy: %w", err)
	}

	return aws.ToString(createResult.Role.Arn), nil
}

func (c *Client) ensureServiceAccount(ctx context.Context, name string) error {
	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: c.namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "slemify",
			},
		},
	}

	_, err := c.clientset.CoreV1().ServiceAccounts(c.namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			return err
		}
		_, err = c.clientset.CoreV1().ServiceAccounts(c.namespace).Create(ctx, sa, metav1.CreateOptions{})
		return err
	}
	return nil
}

func ensurePodIdentityAssociation(ctx context.Context, client *eks.Client, clusterName, namespace, saName, roleARN string) error {
	// Check if association already exists
	listResult, err := client.ListPodIdentityAssociations(ctx, &eks.ListPodIdentityAssociationsInput{
		ClusterName:    aws.String(clusterName),
		Namespace:      aws.String(namespace),
		ServiceAccount: aws.String(saName),
	})
	if err == nil && len(listResult.Associations) > 0 {
		return nil
	}

	created, err := client.CreatePodIdentityAssociation(ctx, &eks.CreatePodIdentityAssociationInput{
		ClusterName:    aws.String(clusterName),
		Namespace:      aws.String(namespace),
		ServiceAccount: aws.String(saName),
		RoleArn:        aws.String(roleARN),
		Tags: map[string]string{
			"slemify.io/purpose": "data-pipeline",
		},
	})
	if err != nil {
		if strings.Contains(err.Error(), "already exists") {
			return nil
		}
		return fmt.Errorf("creating pod identity association: %w", err)
	}

	// A brand-new association is not immediately usable: the Pod Identity
	// agent on the node learns about it asynchronously, and a Job submitted
	// in the same second starts with no credentials (NoCredentialsError in the
	// data pipeline; the project's first deploy failed, the retry passed).
	// Wait until the control plane reports the association, then give the
	// agent a moment before the first pod asks for credentials.
	if created.Association != nil && created.Association.AssociationId != nil {
		waitForPodIdentityAssociation(ctx, client, clusterName, aws.ToString(created.Association.AssociationId))
	}
	return nil
}

// podIdentitySettle is how long to wait after the association is visible.
// Overridable for tests.
var podIdentitySettle = 15 * time.Second

func waitForPodIdentityAssociation(ctx context.Context, client *eks.Client, clusterName, associationID string) {
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		out, err := client.DescribePodIdentityAssociation(ctx, &eks.DescribePodIdentityAssociationInput{
			ClusterName:   aws.String(clusterName),
			AssociationId: aws.String(associationID),
		})
		if err == nil && out.Association != nil && out.Association.AssociationArn != nil {
			fmt.Printf("  Pod Identity: association ready, waiting %s for the node agent\n", podIdentitySettle)
			select {
			case <-ctx.Done():
			case <-time.After(podIdentitySettle):
			}
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(3 * time.Second):
		}
	}
	fmt.Println("  Pod Identity: association not visible after 2m; continuing")
}

// DetectClusterName extracts the EKS cluster name from the current kubeconfig context.
func DetectClusterName(kubeconfigPath string) (string, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	if kubeconfigPath != "" {
		loadingRules.ExplicitPath = kubeconfigPath
	}
	configOverrides := &clientcmd.ConfigOverrides{}
	kubeConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, configOverrides)

	rawConfig, err := kubeConfig.RawConfig()
	if err != nil {
		return "", fmt.Errorf("loading kubeconfig: %w", err)
	}

	currentContext := rawConfig.CurrentContext
	if currentContext == "" {
		return "", fmt.Errorf("no current context in kubeconfig")
	}

	// EKS context ARNs look like: arn:aws:eks:region:account:cluster/name
	parts := strings.Split(currentContext, "/")
	if len(parts) >= 2 && strings.Contains(currentContext, "eks") {
		return parts[len(parts)-1], nil
	}

	// Fallback: try the context's cluster name
	ctxConfig, ok := rawConfig.Contexts[currentContext]
	if !ok {
		return "", fmt.Errorf("context %q not found", currentContext)
	}

	clusterName := ctxConfig.Cluster
	parts = strings.Split(clusterName, "/")
	if len(parts) >= 2 {
		return parts[len(parts)-1], nil
	}

	return clusterName, nil
}
