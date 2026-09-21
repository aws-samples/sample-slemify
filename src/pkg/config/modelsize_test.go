// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

import (
	"strings"
	"testing"
)

func genConfig(base string) []byte {
	return []byte(`apiVersion: slemify/v1
project:
  name: p
  task: generation
  output_format: free_form
  domain: d
model:
  base: "` + base + `"
data:
  bucket: b
  path: x/
`)
}

// Slemify targets dense models up to 8B and the small-MoE class on CPU. A
// larger dense model still sizes (the top tier holds its file) but Parse warns,
// because decode on CPU will be slow and a small-MoE is the better fit.
func TestDenseModelAboveTargetWarns(t *testing.T) {
	_, warnings, err := Parse(genConfig("codellama/CodeLlama-13b-Instruct-hf"))
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "dense ~13B") {
		t.Fatalf("want one dense-size warning, got %v", warnings)
	}
}

func TestMoEAndSmallDenseDoNotWarn(t *testing.T) {
	for _, base := range []string{"Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-8B", ""} {
		_, warnings, err := Parse(genConfig(base))
		if err != nil {
			t.Fatal(err)
		}
		if len(warnings) != 0 {
			t.Fatalf("%q: unexpected warnings %v", base, warnings)
		}
	}
}

// 70B has no tier of its own (Slemify does not target it), but the size is
// still read from the id so the warning names it instead of sizing it as 7B.
func TestLargeDenseModelWarns(t *testing.T) {
	if got := estimateModelSize("meta-llama/Llama-3.1-70B-Instruct"); got != 70 {
		t.Fatalf("estimateModelSize(70B) = %d, want 70", got)
	}
	_, warnings, err := Parse(genConfig("meta-llama/Llama-3.1-70B-Instruct"))
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "dense ~70B") {
		t.Fatalf("want a dense-size warning for 70B, got %v", warnings)
	}
}
