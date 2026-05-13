// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"fmt"
	"os"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
	"github.com/spf13/cobra"
)

var logsCmd = &cobra.Command{
	Use:   "logs",
	Short: "Stream logs from a pipeline stage",
	RunE: func(cmd *cobra.Command, args []string) error {
		stageName, _ := cmd.Flags().GetString("stage")
		follow, _ := cmd.Flags().GetBool("follow")
		ctx := cmd.Context()

		cfg, _, err := config.Load(cfgFile)
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}

		stage, err := pipeline.ParseStage(stageName)
		if err != nil {
			return err
		}

		// Build label selector for the stage's pods
		selector := fmt.Sprintf("slemify.io/project=%s,slemify.io/stage=%s",
			cfg.Project.Name, stageLabelValue(stage))

		client, err := k8s.NewClient(kubeconfig, namespace)
		if err != nil {
			return fmt.Errorf("connecting to cluster: %w", err)
		}

		fmt.Printf("Streaming logs for %s / %s stage", cfg.Project.Name, stage)
		if follow {
			fmt.Print(" (following)")
		}
		fmt.Println("...")

		return client.StreamPodLogs(ctx, selector, follow, os.Stdout)
	},
}

// stageLabelValue maps pipeline stages to the label values used in K8s manifests.
func stageLabelValue(stage pipeline.Stage) string {
	mapping := map[pipeline.Stage]string{
		pipeline.StageData:     "data",
		pipeline.StageTraining: "training",
		pipeline.StageQuantize: "training", // quantize runs in training context
		pipeline.StageServing:  "serving",
	}
	if v, ok := mapping[stage]; ok {
		return v
	}
	return string(stage)
}

func init() {
	logsCmd.Flags().String("stage", "training", "Pipeline stage to stream logs from")
	logsCmd.Flags().BoolP("follow", "f", false, "Follow log output")
	rootCmd.AddCommand(logsCmd)
}
