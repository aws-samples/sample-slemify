// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

import (
	"strings"
)

// AutoSize is a pure function that maps model size to infrastructure decisions.
// It determines GPU count, instance types, checkpoint frequency, KEDA scaling
// thresholds, and Karpenter NodePool configurations.
func AutoSize(model ModelConfig, data DataConfig, training TrainingConfig) SizedConfig {
	return AutoSizeForTask(model, data, training, TaskGeneration)
}

// AutoSizeForTask maps model/task to infrastructure decisions. Encoder-head
// tasks (classification, etc.) train and serve on CPU — no GPU, no GGUF, no
// generation token budgets.
func AutoSizeForTask(model ModelConfig, data DataConfig, training TrainingConfig, task string) SizedConfig {
	if isEncoderHeadTask(task) {
		return autoSizeEncoderHead(data, training)
	}
	return autoSizeGeneration(model, data, training)
}

// isEncoderHeadTask mirrors ProjectConfig.IsEncoderHead for the sizing helper.
func isEncoderHeadTask(task string) bool {
	switch task {
	case TaskClassification, TaskScoring, TaskExtraction, TaskReranking:
		return true
	}
	return false
}

// autoSizeEncoderHead returns CPU-only sizing for the encoder-head family.
// Training is a frozen-encoder embed + lightweight head fit (CPU, minutes);
// serving is the encoder + head on CPU. No GPU, no quantization, no token budgets.
func autoSizeEncoderHead(data DataConfig, training TrainingConfig) SizedConfig {
	sized := SizedConfig{
		TrainingGPU:       "none (CPU)",
		TrainingInstance:  "Spot CPU (Karpenter selects)",
		InferenceInstance: "Spot CPU (Karpenter selects)",
		InferenceCPU:      "2",
		InferenceMemory:   "4Gi",
		InferenceThreads:  "2",
		Scheduler:         "none",
		KEDAMaxReplicas:   10,
	}
	// Epochs/LR are not used by the head trainer (logistic regression solves
	// directly), but keep sane values for display/repro.
	sized.Epochs = 1
	if training.Epochs > 0 {
		sized.Epochs = training.Epochs
	}
	return sized
}

// autoSizeGeneration is the original generative (causal LM) sizing logic.
func autoSizeGeneration(model ModelConfig, data DataConfig, training TrainingConfig) SizedConfig {
	modelSize := estimateModelSize(model.Base)
	sampleCount := estimateSampleCount(data)

	sized := SizedConfig{
		WarmupRatio:       0.1,
		Scheduler:         "cosine",
		EarlyStopPatience: 2,
		KEDAMaxReplicas:   10,
	}

	// Model size -> resource requirements for inference.
	// The pod expresses CPU/memory needs; Karpenter picks the cheapest instance.
	// Training GPU description is for display/reporting only — actual scheduling
	// uses node affinity on instance-gpu-memory.
	switch {
	case modelSize <= 3:
		sized.TrainingGPU = "NVIDIA GPU (≥16GB)"
		sized.TrainingInstance = "Karpenter selects (g/p family)"
		sized.InferenceInstance = "Spot (Karpenter selects)"
		sized.InferenceCPU = "4"
		sized.InferenceMemory = "6Gi"
		sized.InferenceThreads = "4"
		sized.CheckpointInterval = 500 // steps
	case modelSize <= 5:
		sized.TrainingGPU = "NVIDIA GPU (≥16GB)"
		sized.TrainingInstance = "Karpenter selects (g/p family)"
		sized.InferenceInstance = "Spot (Karpenter selects)"
		sized.InferenceCPU = "4"
		sized.InferenceMemory = "8Gi"
		sized.InferenceThreads = "4"
		sized.CheckpointInterval = 250
	case modelSize <= 8:
		sized.TrainingGPU = "NVIDIA GPU (≥16GB)"
		sized.TrainingInstance = "Karpenter selects (g/p family)"
		sized.InferenceInstance = "Spot (Karpenter selects)"
		sized.InferenceCPU = "8"
		sized.InferenceMemory = "16Gi"
		sized.InferenceThreads = "8"
		sized.CheckpointInterval = 100
	default: // 8B-13B (tool targets ≤10B models)
		sized.TrainingGPU = "NVIDIA GPU (≥16GB)"
		sized.TrainingInstance = "Karpenter selects (g/p family)"
		sized.InferenceInstance = "Spot (Karpenter selects)"
		sized.InferenceCPU = "16"
		sized.InferenceMemory = "24Gi"
		sized.InferenceThreads = "16"
		sized.CheckpointInterval = 50
	}

	// No quantization (F16) needs ~3x more memory than Q4_K_M
	qt := model.QuantizeType()
	if qt == "none" || qt == "f16" {
		switch {
		case modelSize <= 3:
			sized.InferenceMemory = "12Gi"
		case modelSize <= 5:
			sized.InferenceMemory = "16Gi"
		case modelSize <= 8:
			sized.InferenceMemory = "24Gi"
		default:
			sized.InferenceMemory = "40Gi"
		}
	} else if qt == "q8_0" {
		// Q8_0 needs ~2x more memory than Q4_K_M
		switch {
		case modelSize <= 3:
			sized.InferenceMemory = "6Gi"
		case modelSize <= 5:
			sized.InferenceMemory = "10Gi"
		case modelSize <= 8:
			sized.InferenceMemory = "16Gi"
		default:
			sized.InferenceMemory = "28Gi"
		}
	}

	// Dataset size -> epochs
	if sampleCount < 10000 {
		sized.Epochs = 5
	} else {
		sized.Epochs = 3
	}

	// User override takes precedence
	if training.Epochs > 0 {
		sized.Epochs = training.Epochs
		// When user explicitly sets high epochs, increase early stopping patience
		// to let the model train longer before giving up
		if training.Epochs >= 6 {
			sized.EarlyStopPatience = 4
		}
	}

	// Model size -> learning rate
	if modelSize <= 7 {
		sized.LearningRate = 2e-4
	} else {
		sized.LearningRate = 1e-4
	}

	return sized
}

// estimateModelSize returns approximate parameter count in billions
// based on common naming patterns in HuggingFace model IDs.
func estimateModelSize(modelID string) int {
	lower := strings.ToLower(modelID)

	// Check from largest to smallest to avoid "1b" matching inside "13b"
	sizeHints := []struct {
		pattern string
		size    int
	}{
		{"70b", 70},
		{"14b", 14},
		{"13b", 13},
		{"8b", 8},
		{"7b", 7},
		{"4b", 4},
		{"3b", 3},
		{"2b", 2},
		{"1.7b", 2},
		{"1.5b", 2},
		{"1b", 1},
		{"0.6b", 1},
		{"0.5b", 1},
	}

	for _, hint := range sizeHints {
		if strings.Contains(lower, hint.pattern) {
			return hint.size
		}
	}

	// Default assumption: 7B (most common SLM size)
	return 7
}

// estimateSampleCount provides a rough sample count estimate.
// In practice this would be determined by reading the actual data from S3.
// For auto-sizing defaults, we use a heuristic based on config.
func estimateSampleCount(data DataConfig) int {
	// If synthetic pairs are configured, use that as a baseline indicator.
	// Real implementation would count actual records in S3.
	if data.Synthetic.Pairs > 0 {
		return data.Synthetic.Pairs * 3 // rough heuristic: raw data ~2x synthetic
	}
	return 5000 // conservative default
}
