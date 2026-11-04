// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"fmt"
	"sort"
	"strings"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/data"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/spf13/cobra"
)

var dataCmd = &cobra.Command{
	Use:   "data",
	Short: "Review the training data",
	Long:  "Download the training and eval data from S3 and show the record counts, the label mix and one sample per label, for a quick check before training.",
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

		trainRecords := data.ParseJSONL(trainData)
		evalRecords := data.ParseJSONL(evalData)
		byOrigin := func(r data.Record) string { return r["origin"] }
		byLabel := func(r data.Record) string { return data.PrimaryLabel(r["output"]) }

		fmt.Printf("━━━ Training Data Review ━━━\n\n")
		fmt.Printf("  Train records:  %d%s\n", len(trainRecords), originSummary(data.CountBy(trainRecords, byOrigin)))
		fmt.Printf("  Eval records:   %d%s\n", len(evalRecords), originSummary(data.CountBy(evalRecords, byOrigin)))
		fmt.Printf("  Total:          %d\n\n", len(trainRecords)+len(evalRecords))

		// Primary label distribution (first pipe-delimited field), training
		// records only: that is the set the model learns from.
		labelCounts := data.CountBy(trainRecords, byLabel)
		labelSorted := sortedCounts(labelCounts)

		fmt.Printf("  Labels: %d\n", len(labelCounts))
		fmt.Printf("  Distribution (train):\n")
		minPerClass := 50
		var weakLabels []string
		for _, s := range labelSorted {
			pct := float64(s.v) / float64(len(trainRecords)) * 100
			marker := ""
			if s.v < minPerClass {
				marker = " ⚠ LOW"
				weakLabels = append(weakLabels, s.k)
			}
			fmt.Printf("    %-30s %4d (%4.1f%%)%s\n", s.k, s.v, pct, marker)
		}

		// Warn about underrepresented labels
		if len(weakLabels) > 0 {
			fmt.Printf("\n  ⚠ %d label(s) below %d training samples: %s\n", len(weakLabels), minPerClass, strings.Join(weakLabels, ", "))
			fmt.Printf("    Add more raw source data for these labels to improve accuracy.\n")
		} else {
			fmt.Printf("\n  ✓ All labels have %d+ training samples.\n", minPerClass)
		}

		// Show samples, one per label, so a run of one label at the start of
		// the file does not fill every slot.
		picked := data.SampleByLabel(trainRecords, samples)
		fmt.Printf("\n━━━ Samples (one per label) ━━━\n")
		for i, r := range picked {
			fmt.Printf("\n[%d/%d]\n", i+1, len(picked))
			if r["instruction"] != "" {
				fmt.Printf("  Instruction: %s\n", truncate(r["instruction"], 80))
			}
			fmt.Printf("  Input:       %s\n", truncate(r["input"], 120))
			fmt.Printf("  Output:      %s\n", r["output"])
			if r["origin"] != "" {
				fmt.Printf("  Origin:      %s\n", r["origin"])
			}
		}
		fmt.Println()

		return nil
	},
}

type countEntry struct {
	k string
	v int
}

// sortedCounts orders counts by value, highest first, then by key.
func sortedCounts(counts map[string]int) []countEntry {
	out := make([]countEntry, 0, len(counts))
	for k, v := range counts {
		out = append(out, countEntry{k, v})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].v != out[j].v {
			return out[i].v > out[j].v
		}
		return out[i].k < out[j].k
	})
	return out
}

// originSummary formats origin counts as "  (labeled 785)", or "" when no
// record has an origin.
func originSummary(counts map[string]int) string {
	if len(counts) == 0 {
		return ""
	}
	var parts []string
	for _, e := range sortedCounts(counts) {
		parts = append(parts, fmt.Sprintf("%s %d", e.k, e.v))
	}
	return "  (" + strings.Join(parts, ", ") + ")"
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
