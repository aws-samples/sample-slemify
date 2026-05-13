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
	"github.com/spf13/cobra"
)

var reportCmd = &cobra.Command{
	Use:   "report",
	Short: "Show the classification accuracy report for a deployed model",
	Long:  "Downloads the HTML report from S3 (generated during deploy), saves it locally, and opens it in the browser.",
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
