// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"context"

	slog "github.com/aws-samples/sample-slemify/pkg/log"
	"github.com/spf13/cobra"
)

// jsonOutput returns true when the user requested JSON output.
func jsonOutput() bool {
	return outputFormat == "json"
}

var (
	cfgFile       string
	namespace     string
	kubeconfig    string
	imageRegistry string
	outputFormat  string
)

var rootCmd = &cobra.Command{
	Use:   "slemify",
	Short: "Domain-specific AI experts on Kubernetes",
	Long:  "Fine-tune and deploy Small Language Models on Kubernetes. One YAML, one command.",
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		slog.Init(outputFormat)
	},
}

// Execute runs the root command.
func Execute() error {
	return rootCmd.Execute()
}

// ExecuteContext runs the root command with a cancellable context.
func ExecuteContext(ctx context.Context) error {
	return rootCmd.ExecuteContext(ctx)
}

func init() {
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "expert.yaml", "Path to Expert Config file")
	rootCmd.PersistentFlags().StringVar(&namespace, "namespace", "slemify", "Kubernetes namespace for resources")
	rootCmd.PersistentFlags().StringVar(&kubeconfig, "kubeconfig", "", "Path to kubeconfig file (default: ~/.kube/config)")
	rootCmd.PersistentFlags().StringVar(&imageRegistry, "image-registry", "", "Container image registry (auto-detects ECR on EKS clusters)")
	rootCmd.PersistentFlags().StringVarP(&outputFormat, "output", "o", "text", "Output format: text or json")
}
