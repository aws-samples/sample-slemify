// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package pipeline

import (
	"context"
	"testing"
)

// --until STAGE runs through that stage and returns, leaving the later stages
// untouched so a later `deploy --stage <next>` picks up from there. Used to
// prepare workshop accounts part-way (data stage done, training left for
// the attendee).
func TestStopAfterRunsThroughStageOnly(t *testing.T) {
	state := NewState("p")
	r := NewRunner("p", state)
	r.SetStopAfter(StageData)
	ran := map[Stage]bool{}
	for _, s := range StageOrder {
		s := s
		r.RegisterStage(s, func(ctx context.Context) ([]string, error) {
			ran[s] = true
			return []string{"artifact"}, nil
		})
	}
	if err := r.Run(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
	if !ran[StageData] {
		t.Fatal("DATA did not run")
	}
	for _, s := range []Stage{StageTraining, StageQuantize, StageServing} {
		if ran[s] {
			t.Fatalf("%s ran; --until data must stop after DATA", s)
		}
		if st := state.Stages[s].Status; st == StatusCompleted || st == StatusInProgress {
			t.Fatalf("%s status = %q; later stages must be left untouched", s, st)
		}
	}
	if got := state.Stages[StageData].Status; got != StatusCompleted {
		t.Fatalf("DATA status = %q, want %q", got, StatusCompleted)
	}
}

// When the stop stage was already completed on a previous run, --until must
// still return there instead of continuing into the later stages.
func TestStopAfterHonoursSkippedStage(t *testing.T) {
	state := NewState("p")
	state.Stages[StageData] = StageResult{Stage: StageData, Status: StatusCompleted, Artifacts: []string{"a"}}
	r := NewRunner("p", state)
	r.SetStopAfter(StageData)
	trainingRan := false
	r.RegisterStage(StageTraining, func(ctx context.Context) ([]string, error) {
		trainingRan = true
		return []string{"m"}, nil
	})
	if err := r.Run(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
	if trainingRan {
		t.Fatal("TRAINING ran after an already-completed --until stage")
	}
}

func TestStageIndex(t *testing.T) {
	if StageIndex(StageData) != 0 || StageIndex(StageServing) != 3 {
		t.Fatalf("unexpected order: data=%d serving=%d", StageIndex(StageData), StageIndex(StageServing))
	}
	if StageIndex(Stage("NOPE")) != -1 {
		t.Fatal("unknown stage should be -1")
	}
}
