// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

import "testing"

func TestEstimateModelSizeMoE(t *testing.T) {
	cases := map[string]int{
		"Qwen/Qwen3-30B-A3B-Instruct-2507": 30, // MoE: total, not the "3b" inside "A3B"
		"Qwen/Qwen3-8B":                    8,
		"meta-llama/Llama-3.2-3B-Instruct": 3,
		"Qwen/Qwen2.5-32B-Instruct":        32,
		"mistralai/Mistral-7B-v0.3":        7,
	}
	for id, want := range cases {
		if got := estimateModelSize(id); got != want {
			t.Errorf("estimateModelSize(%q) = %d, want %d", id, got, want)
		}
	}
}

// The default generation model must get a convert Job that can hold its
// download: 40Gi of ephemeral storage evicted the pod at 61 GB downloaded.
func TestDefaultGenerationConvertSizing(t *testing.T) {
	sized := AutoSizeForTask(ModelConfig{Base: DefaultGenerationBase, Quantize: "q4_k_m"}, DataConfig{}, TrainingConfig{}, "generation")
	if sized.ConvertEphemeralStorage != "160Gi" {
		t.Errorf("ConvertEphemeralStorage = %s, want 160Gi for %s", sized.ConvertEphemeralStorage, DefaultGenerationBase)
	}
	if sized.ConvertMemory != "64Gi" {
		t.Errorf("ConvertMemory = %s, want 64Gi", sized.ConvertMemory)
	}
}
