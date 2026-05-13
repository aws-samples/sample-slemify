// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/spf13/cobra"
)

type validateResult struct {
	Valid    bool                     `json:"valid"`
	Warnings []string                `json:"warnings,omitempty"`
	Errors   []config.ValidationError `json:"errors,omitempty"`
	Sized    *sizedResult            `json:"auto_sized,omitempty"`
}

type sizedResult struct {
	TrainingGPU        string  `json:"training_gpu"`
	TrainingInstance   string  `json:"training_instance"`
	InferenceInstance  string  `json:"inference_instance"`
	InferenceCPU       string  `json:"inference_cpu"`
	InferenceMemory    string  `json:"inference_memory"`
	CheckpointInterval int     `json:"checkpoint_interval"`
	Epochs             int     `json:"epochs"`
	LearningRate       float64 `json:"learning_rate"`
	WarmupRatio        float64 `json:"warmup_ratio"`
	Scheduler          string  `json:"scheduler"`
}

var validateCmd = &cobra.Command{
	Use:   "validate",
	Short: "Validate an Expert Config file and show auto-sized values",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, warnings, err := config.Load(cfgFile)
		if err != nil {
			if jsonOutput() {
				r := validateResult{Valid: false, Errors: []config.ValidationError{{Message: err.Error()}}}
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(r)
			}
			return fmt.Errorf("failed to load config: %w", err)
		}

		errs := config.Validate(cfg)
		sized := config.AutoSize(cfg.Model, cfg.Data, cfg.Training)

		if jsonOutput() {
			r := validateResult{
				Valid:    len(errs) == 0,
				Warnings: warnings,
				Errors:   errs,
				Sized: &sizedResult{
					TrainingGPU:        sized.TrainingGPU,
					TrainingInstance:   sized.TrainingInstance,
					InferenceInstance:  sized.InferenceInstance,
					InferenceCPU:       sized.InferenceCPU,
					InferenceMemory:    sized.InferenceMemory,
					CheckpointInterval: sized.CheckpointInterval,
					Epochs:             sized.Epochs,
					LearningRate:       sized.LearningRate,
					WarmupRatio:        sized.WarmupRatio,
					Scheduler:          sized.Scheduler,
				},
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(r); err != nil {
				return err
			}
			if !r.Valid {
				os.Exit(2)
			}
			return nil
		}

		// Text output
		for _, w := range warnings {
			fmt.Printf("⚠ Warning: %s\n", w)
		}

		if len(errs) > 0 {
			fmt.Println("Validation errors:")
			for _, e := range errs {
				fmt.Printf("  ✗ %s: %s\n", e.Field, e.Message)
			}
			return fmt.Errorf("config has %d validation error(s)", len(errs))
		}

		fmt.Println("✓ Config is valid.")
		fmt.Println()
		fmt.Println("Auto-sized values:")
		fmt.Printf("  Training GPU:         %s\n", sized.TrainingGPU)
		fmt.Printf("  Training instance:    %s\n", sized.TrainingInstance)
		fmt.Printf("  Inference instance:   %s\n", sized.InferenceInstance)
		fmt.Printf("  Checkpoint interval:  every %d steps\n", sized.CheckpointInterval)
		fmt.Printf("  Epochs:               %d\n", sized.Epochs)
		fmt.Printf("  Learning rate:        %g\n", sized.LearningRate)
		fmt.Printf("  Warmup ratio:         %g\n", sized.WarmupRatio)
		fmt.Printf("  LR scheduler:         %s\n", sized.Scheduler)
		fmt.Printf("  Early stop patience:  %d evals\n", sized.EarlyStopPatience)

		return nil
	},
}

func init() {
	rootCmd.AddCommand(validateCmd)
}
