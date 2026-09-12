// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/aws-samples/sample-slemify/pkg/report"
	"github.com/spf13/cobra"
)

var reportCmd = &cobra.Command{
	Use:   "report",
	Short: "Show the report for a deployed model",
	Long: `Prints the summary of the report the deploy step generated and downloads the
full HTML report from S3. Works for every task family: classifiers and
extractors report accuracy against a majority-class baseline, embedding models
report recall against the stock encoder, and generation models report a
serving profile (decode speed, time to first token, bandwidth ceiling).

Optional sections cost frontier-model calls and are off unless set in
expert.yaml under report: (llm_baseline, cases, repeat).`,
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		cfg, _, err := config.Load(cfgFile)
		if err != nil {
			return fmt.Errorf("loading config: %w", err)
		}

		outputFile, _ := cmd.Flags().GetString("output")
		if outputFile == "" {
			outputFile = "report.html"
		}
		noOpen, _ := cmd.Flags().GetBool("no-open")

		client, err := k8s.NewClient(kubeconfig, namespace)
		if err != nil {
			return fmt.Errorf("creating K8s client: %w", err)
		}

		// Terminal summary first; the HTML is the long form of the same data.
		summaryKey := fmt.Sprintf("%s/report/report.json", cfg.Project.Name)
		if data, err := client.DownloadFromS3(ctx, cfg.Data.Bucket, summaryKey); err == nil {
			if s, err := report.ParseSummary(data); err == nil {
				report.PrintSummary(s)
			}
		}

		// Download existing report from S3
		reportKey := fmt.Sprintf("%s/report/report.html", cfg.Project.Name)
		fmt.Printf("Downloading report from s3://%s/%s\n", cfg.Data.Bucket, reportKey)

		reportData, err := client.DownloadFromS3(ctx, cfg.Data.Bucket, reportKey)
		if err != nil {
			return fmt.Errorf("no report found — run 'slemify deploy' first to generate one: %w", err)
		}

		if err := os.WriteFile(outputFile, []byte(reportData), 0644); err != nil {
			return fmt.Errorf("writing report: %w", err)
		}
		fmt.Printf("Report saved to %s\n", outputFile)

		// Open in browser unless --no-open is set
		if !noOpen {
			absPath, _ := filepath.Abs(outputFile)
			if err := openBrowser(absPath); err != nil {
				fmt.Printf("Could not open browser: %v\nOpen %s manually.\n", err, outputFile)
			}
		}

		return nil
	},
}

// openBrowser opens the given file or URL in the default browser.
func openBrowser(path string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", path)
	case "linux":
		cmd = exec.Command("xdg-open", path)
	default:
		return fmt.Errorf("unsupported platform %s", runtime.GOOS)
	}
	return cmd.Start()
}

func init() {
	reportCmd.Flags().String("output", "", "Save HTML report to file (default: report.html)")
	reportCmd.Flags().Bool("no-open", false, "Don't open the report in the browser")
	rootCmd.AddCommand(reportCmd)
}
