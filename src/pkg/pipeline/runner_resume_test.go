// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package pipeline

import (
	"context"
	"testing"
)

// noWaitTrainingState is the saved state after `deploy --stage training
// --no-wait` on a project that was already served once.
func noWaitTrainingState() *State {
	state := NewState("p")
	state.Stages[StageData] = StageResult{Stage: StageData, Status: StatusCompleted, Artifacts: []string{"s3://b/p/processed/train.jsonl"}}
	state.Stages[StageTraining] = StageResult{Stage: StageTraining, Status: StatusInProgress, Artifacts: []string{"job/p-training submitted"}}
	state.Stages[StageServing] = StageResult{Stage: StageServing, Status: StatusCompleted, Artifacts: []string{"deployment/p-inference"}}
	return state
}

func noopStage(ctx context.Context) ([]string, error) { return nil, nil }

// A plain deploy after a --no-wait training tells the training stage it is
// resuming that submission. The stage's output then restarts serving (F3).
func TestPlainRunResumesNoWaitSubmission(t *testing.T) {
	state := noWaitTrainingState()
	r := NewRunner("p", state)
	resuming := false
	served := false
	r.RegisterStage(StageData, func(ctx context.Context) ([]string, error) {
		t.Fatal("completed DATA stage ran again")
		return nil, nil
	})
	r.RegisterStage(StageTraining, func(ctx context.Context) ([]string, error) {
		resuming = ResumingSubmission(ctx)
		return []string{"s3://b/models/p/head.json"}, nil
	})
	r.RegisterStage(StageQuantize, noopStage)
	r.RegisterStage(StageServing, func(ctx context.Context) ([]string, error) {
		if ResumingSubmission(ctx) {
			t.Error("serving stage was told it is resuming a submission")
		}
		served = true
		return []string{"deployment/p-inference"}, nil
	})
	if err := r.Run(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
	if !resuming {
		t.Fatal("training stage was not told it is resuming the --no-wait submission")
	}
	if got := state.Stages[StageTraining].Status; got != StatusCompleted {
		t.Fatalf("training status = %q, want %q", got, StatusCompleted)
	}
	if !served {
		t.Fatal("serving did not restart after the training stage returned new artifacts")
	}
}

// An explicit --stage training is a fresh run, so the stage trains again.
func TestExplicitStageDoesNotResume(t *testing.T) {
	state := noWaitTrainingState()
	r := NewRunner("p", state)
	resuming := true
	r.RegisterStage(StageTraining, func(ctx context.Context) ([]string, error) {
		resuming = ResumingSubmission(ctx)
		return []string{"s3://b/models/p/head.json"}, nil
	})
	r.RegisterStage(StageQuantize, noopStage)
	r.RegisterStage(StageServing, noopStage)
	if err := r.Run(context.Background(), StageTraining); err != nil {
		t.Fatal(err)
	}
	if resuming {
		t.Fatal("explicit --stage training was told it is resuming a submission")
	}
}

// When an earlier stage produced new artifacts in this run, the saved
// submission no longer matches its input, so the stage is not told to resume.
func TestResumeNotSetAfterEarlierStageRan(t *testing.T) {
	state := noWaitTrainingState()
	state.Stages[StageData] = StageResult{Stage: StageData, Status: StatusPending}
	r := NewRunner("p", state)
	resuming := true
	r.RegisterStage(StageData, func(ctx context.Context) ([]string, error) {
		return []string{"s3://b/p/processed/train.jsonl"}, nil
	})
	r.RegisterStage(StageTraining, func(ctx context.Context) ([]string, error) {
		resuming = ResumingSubmission(ctx)
		return []string{"s3://b/models/p/head.json"}, nil
	})
	r.RegisterStage(StageQuantize, noopStage)
	r.RegisterStage(StageServing, noopStage)
	if err := r.Run(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
	if resuming {
		t.Fatal("training was told it is resuming after DATA produced new artifacts")
	}
}
