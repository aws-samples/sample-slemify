// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package report

// ReportScript returns a minimal bootstrap that reads config from env vars
// and calls the report module bundled in the container image.
func ReportScript(bucket, projectName, mcpEndpoint, bedrockModel string, maxSamples int) string {
	// The actual report logic lives in /app/report.py inside the container.
	// This bootstrap just sets the config and calls it.
	return "#!/usr/bin/env python3\nimport report\nreport.main()\n"
}
