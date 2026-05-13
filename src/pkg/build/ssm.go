// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package build

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/ssm"
	ssmtypes "github.com/aws/aws-sdk-go-v2/service/ssm/types"
)

// SSMClient runs commands on EC2 instances via AWS Systems Manager.
// No SSH keys, no open ports, no public IPs required.
type SSMClient struct {
	client     *ssm.Client
	instanceID string
}

// NewSSMClient creates an SSM client targeting a specific EC2 instance.
func NewSSMClient(ctx context.Context, instanceID string) (*SSMClient, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("loading AWS config: %w", err)
	}

	return &SSMClient{
		client:     ssm.NewFromConfig(cfg),
		instanceID: instanceID,
	}, nil
}

// Run executes a shell command on the instance via SSM SendCommand
// and waits for it to complete. Returns the combined stdout output.
func (c *SSMClient) Run(ctx context.Context, cmd string) (string, error) {
	result, err := c.client.SendCommand(ctx, &ssm.SendCommandInput{
		InstanceIds:  []string{c.instanceID},
		DocumentName: aws.String("AWS-RunShellScript"),
		Parameters: map[string][]string{
			"commands":         {cmd},
			"executionTimeout": {"3600"},
		},
		TimeoutSeconds: aws.Int32(3600),
	})
	if err != nil {
		return "", fmt.Errorf("sending command: %w", err)
	}

	commandID := aws.ToString(result.Command.CommandId)
	return c.waitForCommand(ctx, commandID)
}

// waitForCommand polls until the command completes and returns stdout.
func (c *SSMClient) waitForCommand(ctx context.Context, commandID string) (string, error) {
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}

		inv, err := c.client.GetCommandInvocation(ctx, &ssm.GetCommandInvocationInput{
			CommandId:  aws.String(commandID),
			InstanceId: aws.String(c.instanceID),
		})
		if err != nil {
			// InvocationDoesNotExist means the command hasn't been picked up yet
			if strings.Contains(err.Error(), "InvocationDoesNotExist") {
				time.Sleep(2 * time.Second)
				continue
			}
			return "", fmt.Errorf("getting command invocation: %w", err)
		}

		switch inv.Status {
		case ssmtypes.CommandInvocationStatusSuccess:
			return aws.ToString(inv.StandardOutputContent), nil
		case ssmtypes.CommandInvocationStatusFailed,
			ssmtypes.CommandInvocationStatusTimedOut,
			ssmtypes.CommandInvocationStatusCancelled:
			stderr := aws.ToString(inv.StandardErrorContent)
			stdout := aws.ToString(inv.StandardOutputContent)
			output := stdout
			if stderr != "" {
				output = stdout + "\n" + stderr
			}
			return output, fmt.Errorf("command %s: %s", inv.Status, lastLines(output, 20))
		case ssmtypes.CommandInvocationStatusInProgress,
			ssmtypes.CommandInvocationStatusPending,
			ssmtypes.CommandInvocationStatusDelayed:
			time.Sleep(3 * time.Second)
		default:
			time.Sleep(3 * time.Second)
		}
	}
}

// WaitForSSMReady polls until the instance is registered with SSM and can receive commands.
func WaitForSSMReady(ctx context.Context, instanceID string, timeout time.Duration) error {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return fmt.Errorf("loading AWS config: %w", err)
	}
	client := ssm.NewFromConfig(cfg)

	deadline := time.After(timeout)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-deadline:
			return fmt.Errorf("timeout waiting for SSM agent on %s", instanceID)
		case <-ticker.C:
			info, err := client.DescribeInstanceInformation(ctx, &ssm.DescribeInstanceInformationInput{
				Filters: []ssmtypes.InstanceInformationStringFilter{
					{
						Key:    aws.String("InstanceIds"),
						Values: []string{instanceID},
					},
				},
			})
			if err != nil {
				continue
			}
			for _, inst := range info.InstanceInformationList {
				if aws.ToString(inst.InstanceId) == instanceID && inst.PingStatus == ssmtypes.PingStatusOnline {
					return nil
				}
			}
		}
	}
}
