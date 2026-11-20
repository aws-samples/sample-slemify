// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"testing"

	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

func TestStatusIcon(t *testing.T) {
	cases := []struct {
		status string
		stage  pipeline.Stage
		want   string
	}{
		// A training Job that is still running is work in progress.
		{"running", pipeline.StageTraining, "🔄"},
		{"running", pipeline.StageData, "🔄"},
		{"running", pipeline.StageQuantize, "🔄"},
		// The serving Deployment reports running once its replicas are ready.
		{"running", pipeline.StageServing, "✅"},
		{"completed", pipeline.StageTraining, "✅"},
		{"failed", pipeline.StageTraining, "❌"},
		{"starting", pipeline.StageServing, "🔄"},
		{"pending", pipeline.StageTraining, "⏳"},
	}
	for _, c := range cases {
		if got := statusIcon(c.status, c.stage); got != c.want {
			t.Errorf("statusIcon(%q, %q) = %q, want %q", c.status, c.stage, got, c.want)
		}
	}
}
