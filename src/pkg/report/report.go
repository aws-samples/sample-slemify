// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Package report reads the report the in-cluster Job wrote to S3
// (<project>/report/report.json and report.html) and prints its summary in the
// terminal. The Job (containers/data-pipeline/report.py) does the measuring;
// this package only presents. Every task family produces the same two files,
// so `slemify report` works for every project.
package report

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Summary is the loosely typed report.json. Only the fields the terminal
// summary prints are declared; the HTML carries the rest.
type Summary struct {
	Project     string                 `json:"project"`
	Task        string                 `json:"task"`
	GeneratedAt string                 `json:"generated_at"`
	Metrics     map[string]interface{} `json:"metrics"`
	Latency     map[string]interface{} `json:"latency"`
	LLMBaseline map[string]interface{} `json:"llm_baseline"`
	Serving     map[string]interface{} `json:"serving"`
	Profile     map[string]interface{} `json:"profile"`
	Grounded    map[string]interface{} `json:"grounded_eval"`
	Findings    []string               `json:"findings"`
	Guidance    []string               `json:"guidance"`
}

// ParseSummary parses report.json.
func ParseSummary(data string) (*Summary, error) {
	var s Summary
	if err := json.Unmarshal([]byte(data), &s); err != nil {
		return nil, fmt.Errorf("parsing report.json: %w", err)
	}
	return &s, nil
}

func num(m map[string]interface{}, key string) (float64, bool) {
	if m == nil {
		return 0, false
	}
	v, ok := m[key]
	if !ok || v == nil {
		return 0, false
	}
	switch t := v.(type) {
	case float64:
		return t, true
	case int:
		return float64(t), true
	}
	return 0, false
}

func sub(m map[string]interface{}, key string) map[string]interface{} {
	if m == nil {
		return nil
	}
	if v, ok := m[key].(map[string]interface{}); ok {
		return v
	}
	return nil
}

func pctStr(m map[string]interface{}, key string) string {
	if v, ok := num(m, key); ok {
		return fmt.Sprintf("%.1f%%", v*100)
	}
	return "n/a"
}

func msStr(m map[string]interface{}, key string) string {
	if v, ok := num(m, key); ok {
		return fmt.Sprintf("%.0f ms", v)
	}
	return "n/a"
}

// PrintSummary renders the terminal view of the report.
func PrintSummary(s *Summary) {
	fmt.Printf("\n  ━━━ Report: %s (%s) ━━━\n", s.Project, s.Task)
	switch s.Task {
	case "classification":
		printClassification(s)
	case "embedding":
		printEmbedding(s)
	case "scoring":
		fmt.Printf("  MAE:         %s (baseline, predict the mean: %s)\n", numStr(s.Metrics, "mae"), numStr(s.Metrics, "baseline_mae"))
		fmt.Printf("  R squared:   %s   correlation %s\n", numStr(s.Metrics, "r2"), numStr(s.Metrics, "correlation"))
	case "extraction":
		fmt.Printf("  F1:          %s (baseline: %s)\n", pctStr(s.Metrics, "f1"), pctStr(s.Metrics, "baseline_f1"))
	case "generation":
		printGeneration(s)
	}
	if s.Task != "generation" {
		if p50 := msStr(s.Latency, "p50_ms"); p50 != "n/a" {
			fmt.Printf("  Latency:     p50 %s, p95 %s, measured at the endpoint (%v requests)\n",
				p50, msStr(s.Latency, "p95_ms"), s.Latency["n"])
		}
	}
	if it, ok := s.Serving["instance_type"].(string); ok && it != "" {
		line := fmt.Sprintf("  Node:        %s", it)
		if v, ok := num(s.Serving, "vcpus"); ok {
			line += fmt.Sprintf(" (%d vCPUs)", int(v))
		}
		if v, ok := num(s.Serving, "hourly_usd"); ok {
			line += fmt.Sprintf(", on-demand $%.4f/hour for one node", v)
		}
		fmt.Println(line)
	}
	// Findings are the specific observations; the generic order of
	// investigation stays in the HTML.
	if len(s.Findings) > 0 {
		fmt.Printf("  Check first: %s\n", s.Findings[0])
	}
	fmt.Printf("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
}

func numStr(m map[string]interface{}, key string) string {
	if v, ok := num(m, key); ok {
		return fmt.Sprintf("%.3f", v)
	}
	return "n/a"
}

func printClassification(s *Summary) {
	m := s.Metrics
	total, _ := num(m, "total")
	correct, _ := num(m, "correct")
	fmt.Printf("  Accuracy:    %s (%d/%d) exact-match on the held-out set\n", pctStr(m, "accuracy"), int(correct), int(total))
	if base := sub(m, "baseline"); base != nil {
		fmt.Printf("  Baseline:    %s always predicting '%v'\n", pctStr(base, "accuracy"), base["label"])
	}
	if s.LLMBaseline != nil {
		fmt.Printf("  Frontier:    %s zero-shot on %v samples (%v)\n", pctStr(s.LLMBaseline, "accuracy"),
			s.LLMBaseline["n"], s.LLMBaseline["model"])
	}
	if by := sub(m, "by_origin"); by != nil {
		keys := make([]string, 0, len(by))
		for k := range by {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			g := sub(by, k)
			n, _ := num(g, "n")
			parts = append(parts, fmt.Sprintf("%s %s (%d)", k, pctStr(g, "accuracy"), int(n)))
		}
		fmt.Printf("  By origin:   %s\n", strings.Join(parts, ", "))
	}
	if conf, ok := m["confusions"].([]interface{}); ok && len(conf) > 0 {
		fmt.Printf("  Confusions:  ")
		shown := 0
		for _, c := range conf {
			cm, _ := c.(map[string]interface{})
			if cm == nil {
				continue
			}
			if shown > 0 {
				fmt.Printf("; ")
			}
			n, _ := num(cm, "count")
			fmt.Printf("%v -> %v (%d)", cm["expected"], cm["predicted"], int(n))
			shown++
			if shown == 3 {
				break
			}
		}
		fmt.Println()
	}
}

func printEmbedding(s *Summary) {
	m := s.Metrics
	base, tuned := sub(m, "baseline"), sub(m, "tuned")
	fmt.Printf("  recall@2:    stock %s, tuned %s\n", pctStr(base, "recall@2"), pctStr(tuned, "recall@2"))
	fmt.Printf("  recall@5:    stock %s, tuned %s\n", pctStr(base, "recall@5"), pctStr(tuned, "recall@5"))
	fmt.Printf("  MRR:         stock %s, tuned %s\n", numStr(base, "mrr"), numStr(tuned, "mrr"))
	if by := sub(tuned, "by_origin"); by != nil {
		keys := make([]string, 0, len(by))
		for k := range by {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			g := sub(by, k)
			n, _ := num(g, "eval_queries")
			parts = append(parts, fmt.Sprintf("%s recall@2 %s (%d)", k, pctStr(g, "recall@2"), int(n)))
		}
		fmt.Printf("  By origin:   %s\n", strings.Join(parts, ", "))
	}
}

func printGeneration(s *Summary) {
	pr := s.Profile
	if size, ok := num(pr, "model_size_bytes"); ok {
		fmt.Printf("  Model:       %v, %.2f GB on disk\n", pr["model_path"], size/1e9)
	}
	cold, warm := sub(pr, "cold"), sub(pr, "warm")
	if warm != nil {
		if _, failed := warm["error"]; !failed {
			fmt.Printf("  Decode:      %.1f tokens/s (warm)\n", floatOr(warm, "decode_tok_s"))
			// The warm run reports only the tokens it had to process (the cache
			// served the rest), so the prompt length comes from the cold run.
			fmt.Printf("  TTFT:        cold %s, warm %s (%v-token prompt)\n",
				msStr(cold, "ttft_ms"), msStr(warm, "ttft_ms"), cold["prompt_tokens"])
		}
	}
	if ceil := sub(pr, "ceiling"); ceil != nil {
		if c, ok := num(ceil, "tokens_per_second_ceiling"); ok {
			fmt.Printf("  Estimate:    about %.0f tokens/s from bandwidth (%.0f GB/s share / %.2f GB per token)\n",
				c, floatOr(ceil, "bandwidth_gb_s"), floatOr(ceil, "bytes_per_token_assumed")/1e9)
			if f, ok := num(ceil, "measured_fraction"); ok {
				switch {
				case f > 1:
					fmt.Printf("               measured decode is %.0f%% of the estimate: the estimate is a floor for a pod this size\n", f*100)
				case f < 0.6:
					fmt.Printf("               measured decode is %.0f%% of the estimate: check threads and node sharing first\n", f*100)
				default:
					fmt.Printf("               measured decode is %.0f%% of the estimate: bandwidth-bound\n", f*100)
				}
			}
		}
	}
	if s.Grounded != nil {
		fmt.Printf("  Grounded:    %s of drafts made every required point (%v cases x %v repeats)\n",
			pctStr(s.Grounded, "pass_rate"), s.Grounded["cases"], s.Grounded["repeat"])
	}
}

func floatOr(m map[string]interface{}, key string) float64 {
	v, _ := num(m, key)
	return v
}
