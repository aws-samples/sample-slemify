// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Package report generates classification accuracy and latency reports
// by running eval.jsonl through the deployed inference endpoint as a K8s Job.
package report

import (
	"encoding/json"
	"fmt"
)

// ClassificationReport holds the full evaluation results.
type ClassificationReport struct {
	TotalSamples   int                `json:"total_samples"`
	Correct        int                `json:"correct"`
	Accuracy       float64            `json:"accuracy"`
	LatencyP50MS   float64            `json:"latency_p50_ms"`
	LatencyP95MS   float64            `json:"latency_p95_ms"`
	LatencyP99MS   float64            `json:"latency_p99_ms"`
	LatencyAvgMS   float64            `json:"latency_avg_ms"`
	PerCategory    map[string]CatStat `json:"per_category"`
	CostProjection CostProjection     `json:"cost_projection"`
	Predictions    []PredictionResult `json:"predictions,omitempty"`
}

// CatStat holds per-category accuracy stats.
type CatStat struct {
	Total   int     `json:"total"`
	Correct int     `json:"correct"`
	Acc     float64 `json:"accuracy"`
}

// PredictionResult holds the outcome of a single inference call.
type PredictionResult struct {
	Input          string  `json:"input"`
	ExpectedLabel  string  `json:"expected_label"`
	PredictedLabel string  `json:"predicted_label"`
	Correct        bool    `json:"correct"`
	LatencyMS      float64 `json:"latency_ms"`
}

// CostProjection estimates monthly costs.
type CostProjection struct {
	InferenceMonthlyCost float64 `json:"inference_monthly_cost"`
	RequestsPerSecond    float64 `json:"requests_per_second"`
}

// ParseReport parses a JSON report from the Job output.
func ParseReport(data string) (*ClassificationReport, error) {
	var r ClassificationReport
	if err := json.Unmarshal([]byte(data), &r); err != nil {
		return nil, fmt.Errorf("parsing report JSON: %w", err)
	}
	return &r, nil
}

// PrintReport outputs a human-readable summary.
func PrintReport(r *ClassificationReport) {
	fmt.Printf("\n  ━━━ Classification Report ━━━\n")
	fmt.Printf("  Accuracy:    %.1f%% (%d/%d)\n", r.Accuracy*100, r.Correct, r.TotalSamples)
	fmt.Printf("  Latency:     p50=%.0fms  p95=%.0fms  p99=%.0fms\n",
		r.LatencyP50MS, r.LatencyP95MS, r.LatencyP99MS)
	fmt.Printf("  Throughput:  %.1f req/s (single replica)\n", r.CostProjection.RequestsPerSecond)
	fmt.Printf("  Categories:  %d evaluated\n", len(r.PerCategory))

	// Show worst-performing categories
	type catEntry struct {
		name string
		stat CatStat
	}
	var worst []catEntry
	for name, stat := range r.PerCategory {
		if stat.Total >= 2 {
			worst = append(worst, catEntry{name, stat})
		}
	}
	// Sort by accuracy ascending
	for i := 0; i < len(worst); i++ {
		for j := i + 1; j < len(worst); j++ {
			if worst[j].stat.Acc < worst[i].stat.Acc {
				worst[i], worst[j] = worst[j], worst[i]
			}
		}
	}

	if len(worst) > 0 {
		fmt.Printf("  Weakest categories:\n")
		for i, c := range worst {
			if i >= 5 {
				break
			}
			fmt.Printf("    %-40s %.0f%% (%d/%d)\n", c.name, c.stat.Acc*100, c.stat.Correct, c.stat.Total)
		}
	}

	fmt.Printf("  Cost:        ~$%.0f/mo (fixed, Graviton Spot)\n", r.CostProjection.InferenceMonthlyCost)
	fmt.Printf("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
}

// FormatCost formats a dollar amount for display.
func FormatCost(cost float64) string {
	return fmt.Sprintf("$%.2f", cost)
}
