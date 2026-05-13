// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Package pipeline implements the stage sequencer and state machine
// for the Slemify pipeline: INIT → DATA → TRAINING → QUANTIZE → SERVING → COMPLETE.
package pipeline

import (
	"context"
	"fmt"
	"time"
)

// Stage represents a pipeline stage name.
type Stage string

const (
	StageData     Stage = "DATA"
	StageTraining Stage = "TRAINING"
	StageQuantize Stage = "QUANTIZE"
	StageServing  Stage = "SERVING"
)

// StageOrder defines the execution sequence.
var StageOrder = []Stage{StageData, StageTraining, StageQuantize, StageServing}

// Status represents the status of a pipeline stage.
type Status string

const (
	StatusPending    Status = "pending"
	StatusInProgress Status = "in_progress"
	StatusCompleted  Status = "completed"
	StatusFailed     Status = "failed"
)

// StageResult holds the outcome of a single stage execution.
type StageResult struct {
	Stage       Stage     `json:"stage"`
	Status      Status    `json:"status"`
	StartedAt   time.Time `json:"started_at,omitempty"`
	CompletedAt time.Time `json:"completed_at,omitempty"`
	Artifacts   []string  `json:"artifacts,omitempty"`
	Error       string    `json:"error,omitempty"`
}

// StageFunc is the function signature for a stage executor.
// It receives context and returns artifact paths or an error.
type StageFunc func(ctx context.Context) (artifacts []string, err error)

// Runner orchestrates pipeline stage execution.
type Runner struct {
	project    string
	stages     map[Stage]StageFunc
	state      *State
	store      *StateStore
	noWait     bool
	onProgress func(stage Stage, status Status, msg string)
}

// NewRunner creates a pipeline runner for the given project.
func NewRunner(project string, state *State) *Runner {
	return &Runner{
		project: project,
		stages:  make(map[Stage]StageFunc),
		state:   state,
		onProgress: func(stage Stage, status Status, msg string) {
			fmt.Printf("  [%s] %s: %s\n", stage, status, msg)
		},
	}
}

// RegisterStage registers an executor function for a pipeline stage.
func (r *Runner) RegisterStage(stage Stage, fn StageFunc) {
	r.stages[stage] = fn
}

// SetProgressCallback sets a callback for stage progress updates.
func (r *Runner) SetProgressCallback(fn func(Stage, Status, string)) {
	r.onProgress = fn
}

// SetStateStore attaches an S3-backed state store for persistence.
// When set, pipeline state is saved after each stage transition,
// enabling resume after CLI crashes or Spot interruptions.
func (r *Runner) SetStateStore(store *StateStore) {
	r.store = store
}

// SetNoWait makes the runner execute only the first stage and return.
// Used with --no-wait to submit a Job and exit without blocking.
func (r *Runner) SetNoWait(noWait bool) {
	r.noWait = noWait
}

// persistState saves current state to the store if one is configured.
func (r *Runner) persistState(ctx context.Context) {
	if r.store == nil {
		return
	}
	if err := r.store.Save(ctx, r.state); err != nil {
		r.onProgress(r.state.CurrentStage, StatusInProgress, fmt.Sprintf("warning: failed to persist state: %v", err))
	}
}

// Run executes the pipeline from the given start stage through completion.
// If startStage is empty, starts from the beginning.
// Stages with status "completed" and valid artifacts are skipped.
func (r *Runner) Run(ctx context.Context, startStage Stage) error {
	startIdx := 0
	if startStage != "" {
		for i, s := range StageOrder {
			if s == startStage {
				startIdx = i
				break
			}
		}
	}

	for i := startIdx; i < len(StageOrder); i++ {
		stage := StageOrder[i]

		// Check if stage can be skipped (already completed with valid artifacts)
		if result, ok := r.state.Stages[stage]; ok {
			if result.Status == StatusCompleted && len(result.Artifacts) > 0 {
				r.onProgress(stage, StatusCompleted, "skipped (artifacts exist)")
				continue
			}
		}

		fn, ok := r.stages[stage]
		if !ok {
			return fmt.Errorf("no executor registered for stage %s", stage)
		}

		// Mark in-progress
		r.state.CurrentStage = stage
		r.state.Stages[stage] = StageResult{
			Stage:     stage,
			Status:    StatusInProgress,
			StartedAt: time.Now().UTC(),
		}
		r.persistState(ctx)
		r.onProgress(stage, StatusInProgress, "starting")

		// Execute
		artifacts, err := fn(ctx)
		if err != nil {
			r.state.Stages[stage] = StageResult{
				Stage:       stage,
				Status:      StatusFailed,
				StartedAt:   r.state.Stages[stage].StartedAt,
				CompletedAt: time.Now().UTC(),
				Error:       err.Error(),
			}
			r.persistState(ctx)
			r.onProgress(stage, StatusFailed, err.Error())
			return fmt.Errorf("stage %s failed: %w", stage, err)
		}

		// Mark completed
		r.state.Stages[stage] = StageResult{
			Stage:       stage,
			Status:      StatusCompleted,
			StartedAt:   r.state.Stages[stage].StartedAt,
			CompletedAt: time.Now().UTC(),
			Artifacts:   artifacts,
		}
		r.persistState(ctx)

		duration := r.state.Stages[stage].CompletedAt.Sub(r.state.Stages[stage].StartedAt)
		r.onProgress(stage, StatusCompleted, fmt.Sprintf("done (%s)", duration.Round(time.Second)))

		// In no-wait mode, only run the first stage then stop
		if r.noWait {
			return nil
		}
	}

	return nil
}

// ParseStage converts a string to a Stage, returning an error if invalid.
func ParseStage(s string) (Stage, error) {
	mapping := map[string]Stage{
		"data":     StageData,
		"training": StageTraining,
		"quantize": StageQuantize,
		"serving":  StageServing,
	}
	if stage, ok := mapping[s]; ok {
		return stage, nil
	}
	return "", fmt.Errorf("unknown stage %q (valid: data, training, quantize, serving)", s)
}
