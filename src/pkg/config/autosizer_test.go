// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

import (
	"testing"
)

func baseData(pairs int) DataConfig {
	return DataConfig{
		Bucket: "test-bucket",
		Path:   "data/",
		Synthetic: SyntheticConfig{
			Model: "claude-sonnet",
			Pairs: pairs,
		},
	}
}

func TestAutoSize7BModel(t *testing.T) {
	sized := AutoSize(
		ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		baseData(500),
		TrainingConfig{Spot: true},
	)

	if sized.InferenceCPU != "8" {
		t.Errorf("InferenceCPU = %q, want '8'", sized.InferenceCPU)
	}
	if sized.InferenceMemory != "16Gi" {
		t.Errorf("InferenceMemory = %q, want '16Gi'", sized.InferenceMemory)
	}
	if sized.CheckpointInterval != 100 {
		t.Errorf("CheckpointInterval = %d, want 100", sized.CheckpointInterval)
	}
	if sized.LearningRate != 2e-4 {
		t.Errorf("LearningRate = %g, want 2e-4", sized.LearningRate)
	}
	if sized.WarmupRatio != 0.1 {
		t.Errorf("WarmupRatio = %g, want 0.1", sized.WarmupRatio)
	}
	if sized.Scheduler != "cosine" {
		t.Errorf("Scheduler = %q, want 'cosine'", sized.Scheduler)
	}
	if sized.EarlyStopPatience != 2 {
		t.Errorf("EarlyStopPatience = %d, want 2", sized.EarlyStopPatience)
	}
}

func TestAutoSize1BModel(t *testing.T) {
	sized := AutoSize(
		ModelConfig{Base: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
		baseData(500),
		TrainingConfig{},
	)

	if sized.InferenceCPU != "4" {
		t.Errorf("InferenceCPU = %q, want '4' for 1B model", sized.InferenceCPU)
	}
	if sized.InferenceMemory != "6Gi" {
		t.Errorf("InferenceMemory = %q, want '6Gi' for 1B model", sized.InferenceMemory)
	}
	if sized.CheckpointInterval != 500 {
		t.Errorf("CheckpointInterval = %d, want 500 for small model", sized.CheckpointInterval)
	}
	if sized.LearningRate != 2e-4 {
		t.Errorf("LearningRate = %g, want 2e-4 for ≤7B model", sized.LearningRate)
	}
}

func TestAutoSize13BModel(t *testing.T) {
	sized := AutoSize(
		ModelConfig{Base: "codellama/CodeLlama-13b-Instruct-hf"},
		baseData(500),
		TrainingConfig{},
	)

	if sized.InferenceCPU != "16" {
		t.Errorf("InferenceCPU = %q, want '16' for 13B", sized.InferenceCPU)
	}
	if sized.InferenceMemory != "24Gi" {
		t.Errorf("InferenceMemory = %q, want '24Gi' for 13B", sized.InferenceMemory)
	}
	if sized.LearningRate != 1e-4 {
		t.Errorf("LearningRate = %g, want 1e-4 for >7B model", sized.LearningRate)
	}
}

func TestAutoSize70BModel(t *testing.T) {
	// Tool targets ≤10B models. 70B falls into the >8B default tier.
	// No hardcoded instance types — Karpenter selects.
	sized := AutoSize(
		ModelConfig{Base: "meta-llama/Llama-3.1-70B-Instruct"},
		baseData(500),
		TrainingConfig{},
	)

	if sized.InferenceCPU != "16" {
		t.Errorf("InferenceCPU = %q, want '16' for >8B", sized.InferenceCPU)
	}
	if sized.InferenceMemory != "24Gi" {
		t.Errorf("InferenceMemory = %q, want '24Gi' for >8B", sized.InferenceMemory)
	}
	if sized.CheckpointInterval != 50 {
		t.Errorf("CheckpointInterval = %d, want 50 for >8B model", sized.CheckpointInterval)
	}
}

func TestAutoSizeEpochOverride(t *testing.T) {
	sized := AutoSize(
		ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		baseData(5000), // >10K estimated -> would default to 3 epochs
		TrainingConfig{Epochs: 8},
	)
	if sized.Epochs != 8 {
		t.Errorf("Epochs = %d, want 8 (user override)", sized.Epochs)
	}
	if sized.EarlyStopPatience != 4 {
		t.Errorf("EarlyStopPatience = %d, want 4 (increased for high epoch override)", sized.EarlyStopPatience)
	}
}

func TestAutoSizeEpochsSmallDataset(t *testing.T) {
	// 500 pairs * 3 = 1500 estimated samples -> <10K -> 5 epochs
	sized := AutoSize(
		ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		baseData(500),
		TrainingConfig{},
	)
	if sized.Epochs != 5 {
		t.Errorf("Epochs = %d, want 5 for <10K samples", sized.Epochs)
	}
}

func TestAutoSizeEpochsLargeDataset(t *testing.T) {
	// 5000 pairs * 3 = 15000 estimated samples -> >10K -> 3 epochs
	sized := AutoSize(
		ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		baseData(5000),
		TrainingConfig{},
	)
	if sized.Epochs != 3 {
		t.Errorf("Epochs = %d, want 3 for >10K samples", sized.Epochs)
	}
}

func TestAutoSizeUnknownModelDefaultsTo7B(t *testing.T) {
	sized := AutoSize(
		ModelConfig{Base: "some-custom-model-no-size-hint"},
		baseData(500),
		TrainingConfig{},
	)
	// Should default to 7B behavior — resource-based, not instance-based
	if sized.InferenceCPU != "8" {
		t.Errorf("InferenceCPU = %q, want '8' (7B default)", sized.InferenceCPU)
	}
	if sized.LearningRate != 2e-4 {
		t.Errorf("LearningRate = %g, want 2e-4 (≤7B default)", sized.LearningRate)
	}
}

func TestAutoSizeQwen3_4BModel(t *testing.T) {
	sized := AutoSize(
		ModelConfig{Base: "Qwen/Qwen3-4B"},
		baseData(800),
		TrainingConfig{Spot: true},
	)

	if sized.InferenceCPU != "4" {
		t.Errorf("InferenceCPU = %q, want '4' for 4B model", sized.InferenceCPU)
	}
	if sized.InferenceMemory != "8Gi" {
		t.Errorf("InferenceMemory = %q, want '8Gi' for 4B q4_k_m", sized.InferenceMemory)
	}
	if sized.CheckpointInterval != 250 {
		t.Errorf("CheckpointInterval = %d, want 250 for 4B model", sized.CheckpointInterval)
	}
	if sized.LearningRate != 2e-4 {
		t.Errorf("LearningRate = %g, want 2e-4 for ≤7B model", sized.LearningRate)
	}
}

func TestAutoSizeQwen3_8BModelQ8(t *testing.T) {
	sized := AutoSize(
		ModelConfig{Base: "Qwen/Qwen3-8B", Quantize: "q8_0"},
		baseData(2500),
		TrainingConfig{Spot: true},
	)

	if sized.InferenceCPU != "8" {
		t.Errorf("InferenceCPU = %q, want '8' for 8B model", sized.InferenceCPU)
	}
	if sized.InferenceMemory != "16Gi" {
		t.Errorf("InferenceMemory = %q, want '16Gi' for 8B q8_0", sized.InferenceMemory)
	}
	if sized.CheckpointInterval != 100 {
		t.Errorf("CheckpointInterval = %d, want 100 for 8B model", sized.CheckpointInterval)
	}
	if sized.LearningRate != 1e-4 {
		t.Errorf("LearningRate = %g, want 1e-4 for 8B model", sized.LearningRate)
	}
}

func TestAutoSizeQwen3_0_6BModel(t *testing.T) {
	sized := AutoSize(
		ModelConfig{Base: "Qwen/Qwen3-0.6B"},
		baseData(500),
		TrainingConfig{},
	)

	if sized.InferenceCPU != "4" {
		t.Errorf("InferenceCPU = %q, want '4' for 0.6B model", sized.InferenceCPU)
	}
	if sized.InferenceMemory != "6Gi" {
		t.Errorf("InferenceMemory = %q, want '6Gi' for ≤3B model", sized.InferenceMemory)
	}
}

func TestAutoSizeConstantDefaults(t *testing.T) {
	sized := AutoSize(
		ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		baseData(500),
		TrainingConfig{},
	)
	if sized.WarmupRatio != 0.1 {
		t.Errorf("WarmupRatio = %g, want 0.1", sized.WarmupRatio)
	}
	if sized.Scheduler != "cosine" {
		t.Errorf("Scheduler = %q, want 'cosine'", sized.Scheduler)
	}
	if sized.EarlyStopPatience != 2 {
		t.Errorf("EarlyStopPatience = %d, want 2", sized.EarlyStopPatience)
	}
	if sized.KEDAMaxReplicas != 10 {
		t.Errorf("KEDAMaxReplicas = %d, want 10", sized.KEDAMaxReplicas)
	}
}

func TestAutoSizeClassificationIsCPU(t *testing.T) {
	sized := AutoSizeForTask(
		ModelConfig{Base: "BAAI/bge-base-en-v1.5"},
		baseData(1200),
		TrainingConfig{},
		TaskClassification,
	)
	if sized.TrainingGPU != "none (CPU)" {
		t.Errorf("classification TrainingGPU = %q, want 'none (CPU)'", sized.TrainingGPU)
	}
	if sized.InferenceCPU == "" || sized.InferenceMemory == "" {
		t.Error("classification sizing must set inference CPU and memory")
	}
}

func TestAutoSizeForTaskGenerationMatchesAutoSize(t *testing.T) {
	model := ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"}
	a := AutoSize(model, baseData(500), TrainingConfig{})
	b := AutoSizeForTask(model, baseData(500), TrainingConfig{}, TaskGeneration)
	if a.InferenceCPU != b.InferenceCPU || a.TrainingGPU != b.TrainingGPU {
		t.Error("AutoSize and AutoSizeForTask(generation) should agree")
	}
}
