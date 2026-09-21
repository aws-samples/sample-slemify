// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

import (
	"regexp"
	"strconv"
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
	if task == TaskEmbedding {
		// Contrastive embedding training runs on CPU (backprop through a small
		// encoder is feasible for the sizes Slemify targets). Serving is the
		// fine-tuned encoder via ONNX on CPU.
		return autoSizeEncoderHead(data, training)
	}
	return autoSizeGeneration(model, data, training)
}

// isEncoderHeadTask mirrors ProjectConfig.IsEncoderHead for the sizing helper.
func isEncoderHeadTask(task string) bool {
	switch task {
	case TaskClassification, TaskScoring, TaskExtraction:
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
		TrainingInstance:  "CPU, on-demand (provisioner selects)",
		InferenceInstance: "CPU, on-demand (provisioner selects)",
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
		TrainingGPU: "none (CPU)",
		// The generation "training" stage is the GGUF convert Job, pinned to
		// on-demand (one-shot, bandwidth-heavy; a Spot reclaim would force a
		// full re-download). Inference uses the same on-demand pool.
		TrainingInstance:  "CPU, on-demand (provisioner selects)",
		WarmupRatio:       0.1,
		Scheduler:         "cosine",
		EarlyStopPatience: 2,
		KEDAMaxReplicas:   10,
	}

	// Model size -> resource requirements. Inference sizing drives the serving
	// pod (CPU/memory/threads). Convert sizing drives the CPU GGUF conversion
	// Job: it holds the downloaded weights, the intermediate f16 GGUF, and the
	// quantized output on the node's ephemeral disk, and loads the model into
	// memory during conversion. Ephemeral is kept within the slm pool's node
	// volume (the convert runs on the standard CPU pool, no dedicated infra).
	// There is no GPU and no fine-tuning in this path.
	switch {
	case modelSize <= 3:
		sized.InferenceInstance = "CPU, on-demand (provisioner selects)"
		sized.InferenceCPU = "4"
		sized.InferenceMemory = "6Gi"
		sized.InferenceThreads = "4"
		sized.CheckpointInterval = 500 // steps
		sized.ConvertMemory = "16Gi"
		sized.ConvertEphemeralStorage = "40Gi"
	case modelSize <= 5:
		sized.InferenceInstance = "CPU, on-demand (provisioner selects)"
		sized.InferenceCPU = "4"
		sized.InferenceMemory = "8Gi"
		sized.InferenceThreads = "4"
		sized.CheckpointInterval = 250
		sized.ConvertMemory = "24Gi"
		sized.ConvertEphemeralStorage = "48Gi"
	case modelSize <= 8:
		sized.InferenceInstance = "CPU, on-demand (provisioner selects)"
		sized.InferenceCPU = "8"
		sized.InferenceMemory = "16Gi"
		sized.InferenceThreads = "8"
		sized.CheckpointInterval = 100
		sized.ConvertMemory = "40Gi"
		sized.ConvertEphemeralStorage = "64Gi"
	case modelSize <= 14: // 8B-14B
		sized.InferenceInstance = "CPU, on-demand (provisioner selects)"
		sized.InferenceCPU = "16"
		sized.InferenceMemory = "24Gi"
		sized.InferenceThreads = "16"
		sized.CheckpointInterval = 50
		sized.ConvertMemory = "56Gi"
		sized.ConvertEphemeralStorage = "72Gi"
	default: // 30B-class: the small-MoE family (30B total, ~3B active), the largest
		// size Slemify targets on CPU. Dense models of this size are not a
		// target (Parse warns); the tier holds their file but decode is slow.
		// Conversion holds the bf16 download (~61 GB for 30B), the f16 GGUF of
		// the same size, and the quantized output on the node disk at once.
		// Serving loads only the q4_k_m file (~17 GB) and runs the active
		// experts, so it needs the memory of a 30B file and the threads of a
		// ~3B model; 16 threads is what the reference measurements used.
		sized.InferenceInstance = "CPU, on-demand (provisioner selects)"
		sized.InferenceCPU = "16"
		sized.InferenceMemory = "40Gi"
		sized.InferenceThreads = "16"
		sized.CheckpointInterval = 50
		sized.ConvertMemory = "64Gi"
		sized.ConvertEphemeralStorage = "160Gi"
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

// moeName matches mixture-of-experts ids of the form <total>B-A<active>B, such
// as Qwen3-30B-A3B: 30B parameters on disk and in memory during conversion,
// 3B active per token at inference.
var moeName = regexp.MustCompile(`(\d+)b-a(\d+(?:\.\d+)?)b`)

// isMoE reports whether the model id names a mixture-of-experts model.
func isMoE(modelID string) bool {
	return moeName.MatchString(strings.ToLower(modelID))
}

// maxDenseTarget is the largest dense generation model Slemify targets on
// CPU. Above it, only the small-MoE class (30B total, ~3B active) is a
// target: it decodes at roughly dense-3B speed. See docs/deep-dive/training.md.
const maxDenseTarget = 8

// estimateModelSize returns the approximate total parameter count in
// billions from the HuggingFace model id. For MoE ids it is the total, which
// is what the download, the f16 GGUF, and the quantization have to hold.
func estimateModelSize(modelID string) int {
	lower := strings.ToLower(modelID)

	// MoE first: without this, "30B-A3B" matches the "3b" hint below and the
	// convert Job gets 40Gi of disk for a 61 GB download.
	if m := moeName.FindStringSubmatch(lower); m != nil {
		total, _ := strconv.Atoi(m[1])
		return total
	}
	// Dense sizes Slemify has measured on CPU. Nothing above 32B: a dense
	// model that large decodes too slowly on CPU to be a target, and the
	// small-MoE class covers the quality it would bring.

	// Check from largest to smallest to avoid "1b" matching inside "13b"
	sizeHints := []struct {
		pattern string
		size    int
	}{
		{"32b", 32},
		{"30b", 30},
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
	// Any other "<n>b" in the id (a 70B, a 120B): estimate it so the size
	// warning fires instead of silently sizing it as a 7B.
	if m := denseName.FindStringSubmatch(lower); m != nil {
		if n, err := strconv.ParseFloat(m[1], 64); err == nil && n >= 1 {
			return int(n + 0.5)
		}
	}

	// Default assumption: 7B (most common SLM size)
	return 7
}

// denseName matches a parameter count such as "70b" or "1.5b" in a model id.
var denseName = regexp.MustCompile(`(\d+(?:\.\d+)?)b(?:[^a-z0-9]|$)`)

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
