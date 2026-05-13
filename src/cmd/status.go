// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
	"github.com/spf13/cobra"
)

// statusResult is the JSON-serializable status output.
type statusResult struct {
	Project   string        `json:"project"`
	Domain    string        `json:"domain,omitempty"`
	Model     string        `json:"model,omitempty"`
	Namespace string        `json:"namespace"`
	Stages    []stageResult `json:"stages"`
	Artifacts []artifact    `json:"artifacts,omitempty"`
}

type stageResult struct {
	Name           string `json:"name"`
	Status         string `json:"status"` // pending, running, completed, failed
	Detail         string `json:"detail,omitempty"`
	DurationMin    int    `json:"duration_min,omitempty"`
	FailedAttempts int    `json:"failed_attempts,omitempty"`
	FailReason     string `json:"fail_reason,omitempty"`
	LogTail        string `json:"log_tail,omitempty"`
}

type artifact struct {
	Label   string  `json:"label"`
	Path    string  `json:"path"`
	Exists  bool    `json:"exists"`
	SizeMB  float64 `json:"size_mb,omitempty"`
}

var statusCmd = &cobra.Command{
	Use:   "status [project-name]",
	Short: "Show pipeline status from the cluster",
	Long: `Checks Kubernetes Jobs and Deployments in the namespace to show which
pipeline stages are running, completed, or failed.

Can be used with --config (reads project name and bucket from config) or
by passing the project name directly as an argument.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		var project, bucket, domain, model string

		if len(args) > 0 {
			project = args[0]
			bucket, _ = cmd.Flags().GetString("bucket")
		} else {
			cfg, _, err := config.Load(cfgFile)
			if err != nil {
				return fmt.Errorf("failed to load config (use project name as argument or --config): %w", err)
			}
			project = cfg.Project.Name
			bucket = cfg.Data.Bucket
			domain = cfg.Project.Domain
			model = cfg.Model.Base
		}

		client, err := k8s.NewClient(kubeconfig, namespace)
		if err != nil {
			return fmt.Errorf("connecting to cluster: %w", err)
		}

		ctx := cmd.Context()

		result := statusResult{
			Project:   project,
			Domain:    domain,
			Model:     model,
			Namespace: namespace,
		}

		// Map stage names to K8s resource names
		stageJobs := map[pipeline.Stage]string{
			pipeline.StageData:     project + "-data",
			pipeline.StageTraining: project + "-training",
			pipeline.StageQuantize: project + "-quantize",
		}
		stageDeployments := map[pipeline.Stage]string{
			pipeline.StageServing: project + "-inference",
		}

		for _, stage := range pipeline.StageOrder {
			sr := stageResult{Name: string(stage)}

			if jobName, ok := stageJobs[stage]; ok {
				sr.Status, sr.Detail, sr.DurationMin = getJobStatusStructured(ctx, client, jobName)
				if sr.Status == "running" {
					failedCount, failReason := client.GetFailedPodCount(ctx, jobName)
					if failedCount > 0 {
						sr.FailedAttempts = failedCount
						sr.FailReason = failReason
					}
					logs, _ := client.GetJobPodLogs(ctx, jobName)
					if logs != "" {
						// For training stage, extract progress info
						if stage == pipeline.StageTraining {
							sr.LogTail = extractTrainingProgress(logs)
						} else {
							lines := strings.Split(strings.TrimSpace(logs), "\n")
							start := len(lines) - 5
							if start < 0 {
								start = 0
							}
							sr.LogTail = strings.Join(lines[start:], "\n")
						}
					}
				}
			} else if depName, ok := stageDeployments[stage]; ok {
				sr.Status, sr.Detail = getDeploymentStatusStructured(ctx, client, depName)
			} else {
				sr.Status = "pending"
			}

			result.Stages = append(result.Stages, sr)
		}

		// Check S3 artifacts
		if bucket != "" {
			artifactChecks := []struct {
				label string
				key   string
			}{
				{"Training data", fmt.Sprintf("%s/processed/train.jsonl", project)},
				{"Quantized model", fmt.Sprintf("models/%s/model-q4_k_m.gguf", project)},
				{"Report", fmt.Sprintf("%s/report/report.html", project)},
			}
			for _, ac := range artifactChecks {
				a := artifact{Label: ac.label, Path: fmt.Sprintf("s3://%s/%s", bucket, ac.key)}
				exists, size, err := client.CheckS3Object(ctx, bucket, ac.key)
				if err == nil && exists {
					a.Exists = true
					a.SizeMB = float64(size) / (1024 * 1024)
				}
				result.Artifacts = append(result.Artifacts, a)
			}
		}

		// Render output
		if jsonOutput() {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(result)
		}

		// Text output (original format)
		fmt.Printf("Expert: %s\n", project)
		if domain != "" {
			fmt.Printf("Domain: %s\n", domain)
		}
		if model != "" {
			fmt.Printf("Model:  %s\n", model)
		}
		fmt.Printf("Namespace: %s\n\n", namespace)

		fmt.Println("Pipeline stages:")
		for i, sr := range result.Stages {
			prefix := fmt.Sprintf("  %d. %-10s", i+1, sr.Name)
			icon := statusIcon(sr.Status)
			fmt.Printf("%s — %s %s%s\n", prefix, icon, sr.Status, fmtDetail(sr.Detail))

			if sr.Status == "running" && sr.FailedAttempts > 0 {
				detail := sr.FailReason
				if sr.FailReason == "UnexpectedAdmissionError" {
					detail = "GPU not ready on fresh node (normal — NVIDIA device plugin was still initializing)"
				}
				fmt.Printf("     ⚠ %d failed attempt(s) — %s (retried successfully)\n", sr.FailedAttempts, detail)
			}
			if sr.LogTail != "" {
				for _, line := range strings.Split(sr.LogTail, "\n") {
					fmt.Printf("     │ %s\n", line)
				}
			}
		}

		if len(result.Artifacts) > 0 {
			fmt.Println()
			fmt.Println("Artifacts:")
			for _, a := range result.Artifacts {
				if a.Exists {
					if a.SizeMB > 1 {
						fmt.Printf("  ✅ %s — %s (%.1f MB)\n", a.Label, a.Path, a.SizeMB)
					} else {
						fmt.Printf("  ✅ %s — %s\n", a.Label, a.Path)
					}
				} else {
					fmt.Printf("  ⏳ %s — not yet\n", a.Label)
				}
			}
		}

		return nil
	},
}

func getJobStatusStructured(ctx context.Context, client *k8s.Client, jobName string) (status, detail string, durationMin int) {
	js, err := client.GetJobStatus(ctx, jobName)
	if err != nil {
		return "pending", "", 0
	}
	dur := 0
	if js.Duration > 0 {
		dur = int(js.Duration.Minutes())
	}
	switch js.Phase {
	case "Complete":
		return "completed", "", dur
	case "Failed":
		return "failed", js.Reason, dur
	case "Running":
		return "running", "", dur
	default:
		return strings.ToLower(js.Phase), "", dur
	}
}

func getDeploymentStatusStructured(ctx context.Context, client *k8s.Client, depName string) (string, string) {
	ready, total, err := client.GetDeploymentReadiness(ctx, depName)
	if err != nil {
		return "pending", ""
	}
	if ready >= total && total > 0 {
		return "running", fmt.Sprintf("%d/%d ready", ready, total)
	}
	return "starting", fmt.Sprintf("%d/%d ready", ready, total)
}

func statusIcon(status string) string {
	switch status {
	case "completed":
		return "✅"
	case "running":
		return "✅"
	case "failed":
		return "❌"
	case "starting":
		return "🔄"
	default:
		return "⏳"
	}
}

func fmtDetail(d string) string {
	if d == "" {
		return ""
	}
	return " (" + d + ")"
}

func checkS3Artifact(ctx context.Context, client *k8s.Client, bucket, key, label string) {
	exists, size, err := client.CheckS3Object(ctx, bucket, key)
	if err != nil || !exists {
		fmt.Printf("  ⏳ %s — not yet\n", label)
	} else {
		sizeMB := float64(size) / (1024 * 1024)
		if sizeMB > 1 {
			fmt.Printf("  ✅ %s — s3://%s/%s (%.1f MB)\n", label, bucket, key, sizeMB)
		} else {
			fmt.Printf("  ✅ %s — s3://%s/%s\n", label, bucket, key)
		}
	}
}

func init() {
	statusCmd.Flags().String("bucket", "", "S3 bucket for artifacts (required when using project name argument)")
	rootCmd.AddCommand(statusCmd)
}

// extractTrainingProgress parses Unsloth training logs to extract a concise progress summary.
func extractTrainingProgress(logs string) string {
	// Clean carriage returns (progress bars use \r to overwrite in-place)
	logs = strings.ReplaceAll(logs, "\r", "\n")
	lines := strings.Split(logs, "\n")
	var progress, loss, config string

	for _, line := range lines {
		line = strings.TrimSpace(line)

		// Match progress bar: "16%|█▌ | 250/1565 [17:03<1:28:55"
		if strings.Contains(line, "/") && strings.Contains(line, "|") && strings.Contains(line, "%") {
			progress = line
		}

		// Match loss JSON: {"loss": 1.234, "learning_rate": 0.0002, "epoch": 0.5}
		if strings.HasPrefix(line, "{\"loss\"") {
			loss = line
		}

		// Match training config: "Num examples = 2,500 | Num Epochs = 5 | Total steps = 1,565"
		if strings.Contains(line, "Num examples") && strings.Contains(line, "Total steps") {
			config = line
		}

		// Match GGUF export or completion
		if strings.Contains(line, "Exporting to GGUF") || strings.Contains(line, "GGUF export complete") ||
			strings.Contains(line, "Training complete") || strings.Contains(line, "Upload complete") {
			progress = line
		}
	}

	var parts []string
	if config != "" {
		parts = append(parts, strings.TrimSpace(config))
	}
	if progress != "" {
		parts = append(parts, strings.TrimSpace(progress))
	}
	if loss != "" {
		var lossData map[string]interface{}
		if err := json.Unmarshal([]byte(loss), &lossData); err == nil {
			if l, ok := lossData["loss"].(float64); ok {
				epoch := lossData["epoch"]
				parts = append(parts, fmt.Sprintf("Loss: %.4f (epoch %.1f)", l, epoch))
			}
		}
	}

	if len(parts) == 0 {
		var tail []string
		for i := len(lines) - 1; i >= 0 && len(tail) < 3; i-- {
			if strings.TrimSpace(lines[i]) != "" {
				tail = append([]string{strings.TrimSpace(lines[i])}, tail...)
			}
		}
		return strings.Join(tail, "\n")
	}

	return strings.Join(parts, "\n")
}
