// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var starterConfig = `apiVersion: slemify/v1

# Project identity
project:
  name: my-project                   # DNS-compatible name (lowercase, hyphens)
  task: generation                   # generation | classification | scoring | extraction | reranking | embedding
  domain: "Describe your task here"  # e.g., "Alert triage and severity classification"

# Base model for fine-tuning (HuggingFace model ID)
model:
  base: HuggingFaceTB/SmolLM3-3B    # 3B for classification, 7B for extraction
  quantize: q4_k_m                   # q4_k_m (default), q8_0, f16, none

# Data pipeline configuration
data:
  bucket: my-data-bucket             # S3 bucket for data and artifacts
  path: data/                        # Path prefix in the bucket
  sources:
    - path: training-data/
      type: documentation
      # metadata:
      #   priority: high             # Optional: high, medium, low
  synthetic:
    model: claude-sonnet             # Bedrock model (no endpoint = Bedrock)
    pairs: 500                       # 200-500 for classification, 500-1000 for extraction
    # endpoint: http://localhost:11434  # Ollama or any OpenAI-compatible API

# Training configuration
training:
  spot: true                         # Use Spot instances (60% savings, auto-checkpoint)

# Agent tools exposed via MCP
agent:
  tools:
    - name: classify
      description: "Classify input and return structured JSON output"
`

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Generate a starter config file",
	RunE: func(cmd *cobra.Command, args []string) error {
		output, _ := cmd.Flags().GetString("output")

		if _, err := os.Stat(output); err == nil {
			return fmt.Errorf("%s already exists — use a different name or remove it first", output)
		}

		if err := os.WriteFile(output, []byte(starterConfig), 0644); err != nil {
			return fmt.Errorf("failed to write config: %w", err)
		}

		fmt.Printf("Created %s — edit it with your task details, then run:\n", output)
		fmt.Printf("  slemify validate --config %s\n", output)
		fmt.Printf("  slemify deploy --config %s --dry-run\n", output)
		return nil
	},
}

func init() {
	initCmd.Flags().StringP("output", "o", "expert.yaml", "Output file path")
	rootCmd.AddCommand(initCmd)
}
