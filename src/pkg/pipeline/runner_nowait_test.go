// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package pipeline

import (
	"context"
	"testing"
)

// A --no-wait run submits a job and returns. The stage must be left
// in_progress, not completed, so the next run re-executes it rather than
// skipping a stage whose only "artifact" is a job name.
func TestNoWaitLeavesStageInProgress(t *testing.T) {
	state := NewState("p")
	r := NewRunner("p", state)
	r.SetNoWait(true)
	r.RegisterStage(StageData, func(ctx context.Context) ([]string, error) {
		return []string{"job/p-data submitted"}, nil
	})
	if err := r.Run(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
	if got := state.Stages[StageData].Status; got != StatusInProgress {
		t.Fatalf("after --no-wait: status = %q, want %q", got, StatusInProgress)
	}

	// The next run, without --no-wait, must execute the stage again.
	ran := false
	r2 := NewRunner("p", state)
	r2.RegisterStage(StageData, func(ctx context.Context) ([]string, error) {
		ran = true
		return []string{"s3://b/p/processed/train.jsonl"}, nil
	})
	_ = r2.Run(context.Background(), "") // later stages are unregistered; only the DATA re-run matters
	if !ran {
		t.Fatal("second run skipped the stage that --no-wait left in_progress")
	}
	if got := state.Stages[StageData].Status; got != StatusCompleted {
		t.Fatalf("after the real run: status = %q, want %q", got, StatusCompleted)
	}
}
