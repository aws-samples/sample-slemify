// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package pipeline

import (
	"encoding/json"
	"testing"
)

func TestNewStateInitializesAllStages(t *testing.T) {
	state := NewState("karpenter-expert")

	if state.Project != "karpenter-expert" {
		t.Errorf("Project = %q, want %q", state.Project, "karpenter-expert")
	}

	for _, stage := range StageOrder {
		result, ok := state.Stages[stage]
		if !ok {
			t.Errorf("stage %s not found in state", stage)
			continue
		}
		if result.Status != StatusPending {
			t.Errorf("stage %s status = %s, want %s", stage, result.Status, StatusPending)
		}
		if result.Stage != stage {
			t.Errorf("stage %s result.Stage = %s, want %s", stage, result.Stage, stage)
		}
	}

	if len(state.Stages) != len(StageOrder) {
		t.Errorf("state has %d stages, want %d", len(state.Stages), len(StageOrder))
	}
}

func TestStateJSONRoundTrip(t *testing.T) {
	state := NewState("test-project")
	state.CurrentStage = StageTraining
	state.Stages[StageData] = StageResult{
		Stage:    StageData,
		Status:   StatusCompleted,
		Artifacts: []string{"s3://bucket/train.jsonl", "s3://bucket/eval.jsonl"},
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var restored State
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if restored.Project != state.Project {
		t.Errorf("Project = %q, want %q", restored.Project, state.Project)
	}
	if restored.CurrentStage != state.CurrentStage {
		t.Errorf("CurrentStage = %s, want %s", restored.CurrentStage, state.CurrentStage)
	}

	dataResult := restored.Stages[StageData]
	if dataResult.Status != StatusCompleted {
		t.Errorf("DATA status = %s, want completed", dataResult.Status)
	}
	if len(dataResult.Artifacts) != 2 {
		t.Errorf("DATA artifacts = %d, want 2", len(dataResult.Artifacts))
	}
}

func TestStateKeyFormat(t *testing.T) {
	key := stateKey("karpenter-expert")
	expected := "karpenter-expert/state/pipeline-state.json"
	if key != expected {
		t.Errorf("stateKey = %q, want %q", key, expected)
	}
}
