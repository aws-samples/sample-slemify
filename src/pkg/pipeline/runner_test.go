// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package pipeline

import (
	"context"
	"fmt"
	"testing"
)

// mockStageFunc returns a StageFunc that records execution and returns given artifacts.
func mockStageFunc(executed *[]Stage, stage Stage, artifacts []string) StageFunc {
	return func(ctx context.Context) ([]string, error) {
		*executed = append(*executed, stage)
		return artifacts, nil
	}
}

// failingStageFunc returns a StageFunc that fails with the given error.
func failingStageFunc(executed *[]Stage, stage Stage, errMsg string) StageFunc {
	return func(ctx context.Context) ([]string, error) {
		*executed = append(*executed, stage)
		return nil, fmt.Errorf("%s", errMsg)
	}
}

func registerAllStages(r *Runner, executed *[]Stage) {
	r.RegisterStage(StageData, mockStageFunc(executed, StageData, []string{"s3://bucket/train.jsonl"}))
	r.RegisterStage(StageTraining, mockStageFunc(executed, StageTraining, []string{"s3://bucket/model/"}))
	r.RegisterStage(StageQuantize, mockStageFunc(executed, StageQuantize, []string{"s3://bucket/model.gguf"}))
	r.RegisterStage(StageServing, mockStageFunc(executed, StageServing, []string{"inference.svc:8080"}))
}

func TestRunAllStagesInOrder(t *testing.T) {
	state := NewState("test-project")
	runner := NewRunner("test-project", state)
	runner.SetProgressCallback(func(Stage, Status, string) {}) // silence output

	var executed []Stage
	registerAllStages(runner, &executed)

	err := runner.Run(context.Background(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []Stage{StageData, StageTraining, StageQuantize, StageServing}
	if len(executed) != len(expected) {
		t.Fatalf("executed %d stages, want %d", len(executed), len(expected))
	}
	for i, s := range expected {
		if executed[i] != s {
			t.Errorf("stage %d = %s, want %s", i, executed[i], s)
		}
	}
}

func TestCompletedStagesAreSkipped(t *testing.T) {
	state := NewState("test-project")
	// Mark DATA as completed with artifacts
	state.Stages[StageData] = StageResult{
		Stage:    StageData,
		Status:   StatusCompleted,
		Artifacts: []string{"s3://bucket/train.jsonl"},
	}

	runner := NewRunner("test-project", state)
	runner.SetProgressCallback(func(Stage, Status, string) {})

	var executed []Stage
	registerAllStages(runner, &executed)

	err := runner.Run(context.Background(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// DATA should be skipped
	for _, s := range executed {
		if s == StageData {
			t.Error("DATA stage should have been skipped (already completed)")
		}
	}
	if len(executed) != 3 {
		t.Errorf("executed %d stages, want 3 (DATA skipped)", len(executed))
	}
}

func TestCompletedWithoutArtifactsReexecutes(t *testing.T) {
	state := NewState("test-project")
	// Completed but no artifacts — should re-execute
	state.Stages[StageData] = StageResult{
		Stage:  StageData,
		Status: StatusCompleted,
	}

	runner := NewRunner("test-project", state)
	runner.SetProgressCallback(func(Stage, Status, string) {})

	var executed []Stage
	registerAllStages(runner, &executed)

	err := runner.Run(context.Background(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// DATA should re-execute since it has no artifacts
	if len(executed) != 4 {
		t.Errorf("executed %d stages, want 4 (all stages)", len(executed))
	}
}

func TestFailedStageStopsExecution(t *testing.T) {
	state := NewState("test-project")
	runner := NewRunner("test-project", state)
	runner.SetProgressCallback(func(Stage, Status, string) {})

	var executed []Stage
	runner.RegisterStage(StageData, mockStageFunc(&executed, StageData, []string{"data"}))
	runner.RegisterStage(StageTraining, failingStageFunc(&executed, StageTraining, "GPU OOM"))
	runner.RegisterStage(StageQuantize, mockStageFunc(&executed, StageQuantize, []string{"model"}))
	runner.RegisterStage(StageServing, mockStageFunc(&executed, StageServing, []string{"svc"}))

	err := runner.Run(context.Background(), "")
	if err == nil {
		t.Fatal("expected error from failed stage")
	}

	// Only DATA and TRAINING should have executed
	if len(executed) != 2 {
		t.Errorf("executed %d stages, want 2 (stopped at TRAINING)", len(executed))
	}

	// State should reflect the failure
	result := state.Stages[StageTraining]
	if result.Status != StatusFailed {
		t.Errorf("TRAINING status = %s, want %s", result.Status, StatusFailed)
	}
	if result.Error != "GPU OOM" {
		t.Errorf("TRAINING error = %q, want %q", result.Error, "GPU OOM")
	}

	// QUANTIZE should still be pending
	if state.Stages[StageQuantize].Status != StatusPending {
		t.Errorf("QUANTIZE status = %s, want %s", state.Stages[StageQuantize].Status, StatusPending)
	}
}

func TestResumeFromStage(t *testing.T) {
	state := NewState("test-project")
	runner := NewRunner("test-project", state)
	runner.SetProgressCallback(func(Stage, Status, string) {})

	var executed []Stage
	registerAllStages(runner, &executed)

	err := runner.Run(context.Background(), StageQuantize)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should only execute QUANTIZE, SERVING
	expected := []Stage{StageQuantize, StageServing}
	if len(executed) != len(expected) {
		t.Fatalf("executed %d stages, want %d", len(executed), len(expected))
	}
	for i, s := range expected {
		if executed[i] != s {
			t.Errorf("stage %d = %s, want %s", i, executed[i], s)
		}
	}
}

func TestExplicitStartStageForcesRerun(t *testing.T) {
	// Even if DATA is marked completed with artifacts, explicitly starting from
	// DATA must re-run it (stale state may reference removed/regenerated artifacts).
	state := NewState("test-project")
	state.Stages[StageData] = StageResult{
		Stage:     StageData,
		Status:    StatusCompleted,
		Artifacts: []string{"s3://bucket/train.jsonl"},
	}

	runner := NewRunner("test-project", state)
	runner.SetProgressCallback(func(Stage, Status, string) {})

	var executed []Stage
	registerAllStages(runner, &executed)

	if err := runner.Run(context.Background(), StageData); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(executed) == 0 || executed[0] != StageData {
		t.Errorf("explicit start at DATA should re-run DATA, executed: %v", executed)
	}
}

func TestUnregisteredStageReturnsError(t *testing.T) {
	state := NewState("test-project")
	runner := NewRunner("test-project", state)
	runner.SetProgressCallback(func(Stage, Status, string) {})

	// Only register DATA, skip the rest
	var executed []Stage
	runner.RegisterStage(StageData, mockStageFunc(&executed, StageData, []string{"data"}))

	err := runner.Run(context.Background(), "")
	if err == nil {
		t.Fatal("expected error for unregistered stage")
	}
}

func TestStateTracksTimestampsAndArtifacts(t *testing.T) {
	state := NewState("test-project")
	runner := NewRunner("test-project", state)
	runner.SetProgressCallback(func(Stage, Status, string) {})

	var executed []Stage
	registerAllStages(runner, &executed)

	err := runner.Run(context.Background(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, stage := range StageOrder {
		result := state.Stages[stage]
		if result.Status != StatusCompleted {
			t.Errorf("%s status = %s, want completed", stage, result.Status)
		}
		if result.StartedAt.IsZero() {
			t.Errorf("%s StartedAt is zero", stage)
		}
		if result.CompletedAt.IsZero() {
			t.Errorf("%s CompletedAt is zero", stage)
		}
		if !result.CompletedAt.After(result.StartedAt) && !result.CompletedAt.Equal(result.StartedAt) {
			t.Errorf("%s CompletedAt should be >= StartedAt", stage)
		}
		if len(result.Artifacts) == 0 {
			t.Errorf("%s should have artifacts", stage)
		}
	}
}

func TestParseStageValid(t *testing.T) {
	cases := map[string]Stage{
		"data":     StageData,
		"training": StageTraining,
		"quantize": StageQuantize,
		"serving":  StageServing,
	}
	for input, expected := range cases {
		stage, err := ParseStage(input)
		if err != nil {
			t.Errorf("ParseStage(%q) error: %v", input, err)
		}
		if stage != expected {
			t.Errorf("ParseStage(%q) = %s, want %s", input, stage, expected)
		}
	}
}

func TestParseStageInvalid(t *testing.T) {
	_, err := ParseStage("invalid")
	if err == nil {
		t.Error("expected error for invalid stage name")
	}
}

func TestContextCancellation(t *testing.T) {
	state := NewState("test-project")
	runner := NewRunner("test-project", state)
	runner.SetProgressCallback(func(Stage, Status, string) {})

	ctx, cancel := context.WithCancel(context.Background())

	var executed []Stage
	runner.RegisterStage(StageData, func(ctx context.Context) ([]string, error) {
		executed = append(executed, StageData)
		cancel() // cancel after DATA
		return []string{"data"}, nil
	})
	runner.RegisterStage(StageTraining, func(ctx context.Context) ([]string, error) {
		// Check if context is cancelled
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		executed = append(executed, StageTraining)
		return []string{"model"}, nil
	})
	runner.RegisterStage(StageQuantize, mockStageFunc(&executed, StageQuantize, []string{"gguf"}))
	runner.RegisterStage(StageServing, mockStageFunc(&executed, StageServing, []string{"svc"}))

	err := runner.Run(ctx, "")
	if err == nil {
		t.Fatal("expected error from context cancellation")
	}
}
