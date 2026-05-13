// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/spf13/cobra"
)

var dataCmd = &cobra.Command{
	Use:   "data",
	Short: "Review generated training data",
	Long:  "Download and display training data samples from S3 for quick validation before training.",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		cfg, _, err := config.Load(cfgFile)
		if err != nil {
			return fmt.Errorf("loading config: %w", err)
		}

		samples, _ := cmd.Flags().GetInt("samples")

		client, err := k8s.NewClient(kubeconfig, namespace)
		if err != nil {
			return fmt.Errorf("creating client: %w", err)
		}

		// Download train and eval
		trainKey := fmt.Sprintf("%s/processed/train.jsonl", cfg.Project.Name)
		evalKey := fmt.Sprintf("%s/processed/eval.jsonl", cfg.Project.Name)

		trainData, err := client.DownloadFromS3(ctx, cfg.Data.Bucket, trainKey)
		if err != nil {
			return fmt.Errorf("no training data found — run 'slemify deploy --stage data' first: %w", err)
		}
		evalData, _ := client.DownloadFromS3(ctx, cfg.Data.Bucket, evalKey)

		trainRecords := parseJSONL(trainData)
		evalRecords := parseJSONL(evalData)

		fmt.Printf("━━━ Training Data Review ━━━\n\n")
		fmt.Printf("  Train records:  %d\n", len(trainRecords))
		fmt.Printf("  Eval records:   %d\n", len(evalRecords))
		fmt.Printf("  Total:          %d\n\n", len(trainRecords)+len(evalRecords))

		// Output distribution
		// Primary label distribution (first pipe-delimited field)
		labelCounts := make(map[string]int)
		for _, r := range append(trainRecords, evalRecords...) {
			output := r["output"]
			label := output
			if idx := strings.Index(output, "|"); idx > 0 {
				label = output[:idx]
			}
			labelCounts[strings.TrimSpace(label)]++
		}

		type kv struct {
			k string
			v int
		}
		var labelSorted []kv
		for k, v := range labelCounts {
			labelSorted = append(labelSorted, kv{k, v})
		}
		for i := 0; i < len(labelSorted); i++ {
			for j := i + 1; j < len(labelSorted); j++ {
				if labelSorted[j].v > labelSorted[i].v {
					labelSorted[i], labelSorted[j] = labelSorted[j], labelSorted[i]
				}
			}
		}

		fmt.Printf("  Labels: %d\n", len(labelCounts))
		fmt.Printf("  Distribution:\n")
		minPerClass := 50
		for _, s := range labelSorted {
			pct := float64(s.v) / float64(len(trainRecords)+len(evalRecords)) * 100
			marker := ""
			if s.v < minPerClass {
				marker = " ⚠ LOW"
			}
			fmt.Printf("    %-30s %4d (%4.1f%%)%s\n", s.k, s.v, pct, marker)
		}

		// Warn about underrepresented labels
		weak := 0
		var weakLabels []string
		for _, s := range labelSorted {
			if s.v < minPerClass {
				weak++
				weakLabels = append(weakLabels, s.k)
			}
		}
		if weak > 0 {
			fmt.Printf("\n  ⚠ %d label(s) below %d samples: %s\n", weak, minPerClass, strings.Join(weakLabels, ", "))
			fmt.Printf("    Add more raw source data for these labels to improve accuracy.\n")
		} else {
			fmt.Printf("\n  ✓ All labels have %d+ samples.\n", minPerClass)
		}

		// Show samples
		fmt.Printf("\n━━━ Samples ━━━\n")
		show := samples
		if show > len(trainRecords) {
			show = len(trainRecords)
		}
		for i := 0; i < show; i++ {
			r := trainRecords[i]
			fmt.Printf("\n[%d/%d]\n", i+1, show)
			fmt.Printf("  Instruction: %s\n", truncate(r["instruction"], 80))
			fmt.Printf("  Input:       %s\n", truncate(r["input"], 120))
			fmt.Printf("  Output:      %s\n", r["output"])
		}
		fmt.Println()

		return nil
	},
}

func parseJSONL(data string) []map[string]string {
	var records []map[string]string
	for _, line := range strings.Split(data, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var r map[string]string
		if err := json.Unmarshal([]byte(line), &r); err == nil {
			records = append(records, r)
		}
	}
	return records
}

func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}

func init() {
	dataCmd.Flags().Int("samples", 5, "Number of sample records to display")
	rootCmd.AddCommand(dataCmd)
}
