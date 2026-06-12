// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

import (
	"testing"
)

// validKarpenterYAML is the reference Karpenter SME config used across tests.
const validKarpenterYAML = `
apiVersion: slemify/v1
project:
  name: karpenter-expert
  task: generation
  domain: "Karpenter configuration and optimization on EKS"
  domain_version: "1.2"
model:
  base: mistralai/Mistral-7B-Instruct-v0.3
data:
  bucket: my-bucket
  path: karpenter-data/
  sources:
    - path: github-issues/
      type: github-issues
    - path: docs/
      type: documentation
      metadata:
        priority: high
  synthetic:
    model: claude-sonnet
    pairs: 500
training:
  spot: true
`

// --- Parse Tests ---

func TestParseValidConfig(t *testing.T) {
	cfg, warnings, err := Parse([]byte(validKarpenterYAML))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(warnings) > 0 {
		t.Fatalf("unexpected warnings: %v", warnings)
	}

	if cfg.APIVersion != "slemify/v1" {
		t.Errorf("apiVersion = %q, want %q", cfg.APIVersion, "slemify/v1")
	}
	if cfg.Project.Name != "karpenter-expert" {
		t.Errorf("project.name = %q, want %q", cfg.Project.Name, "karpenter-expert")
	}
	if cfg.Project.DomainVersion != "1.2" {
		t.Errorf("project.domain_version = %q, want %q", cfg.Project.DomainVersion, "1.2")
	}
	if cfg.Model.Base != "mistralai/Mistral-7B-Instruct-v0.3" {
		t.Errorf("model.base = %q, want mistralai/Mistral-7B-Instruct-v0.3", cfg.Model.Base)
	}
	if cfg.Data.Bucket != "my-bucket" {
		t.Errorf("data.bucket = %q, want %q", cfg.Data.Bucket, "my-bucket")
	}
	if len(cfg.Data.Sources) != 2 {
		t.Fatalf("data.sources length = %d, want 2", len(cfg.Data.Sources))
	}
	if cfg.Data.Sources[0].Type != "github-issues" {
		t.Errorf("sources[0].type = %q, want %q", cfg.Data.Sources[0].Type, "github-issues")
	}
	if cfg.Data.Sources[1].Metadata == nil || cfg.Data.Sources[1].Metadata.Priority != "high" {
		t.Errorf("sources[1].metadata.priority should be 'high'")
	}
	if cfg.Data.Synthetic.Model != "claude-sonnet" {
		t.Errorf("synthetic.model = %q, want %q", cfg.Data.Synthetic.Model, "claude-sonnet")
	}
	if cfg.Data.Synthetic.Pairs != 500 {
		t.Errorf("synthetic.pairs = %d, want 500", cfg.Data.Synthetic.Pairs)
	}
	if cfg.Data.Synthetic.Endpoint != "" {
		t.Errorf("synthetic.endpoint = %q, want empty (Bedrock default)", cfg.Data.Synthetic.Endpoint)
	}
	if !cfg.Training.Spot {
		t.Error("training.spot should be true")
	}
}

func TestParseWithOpenAIEndpoint(t *testing.T) {
	yaml := `
apiVersion: slemify/v1
project:
  name: test-expert
  task: generation
  domain: "test domain"
model:
  base: meta-llama/Llama-3.1-8B-Instruct
data:
  bucket: test-bucket
  path: data/
  synthetic:
    model: llama3.1:8b
    endpoint: http://ollama.internal:11434
    pairs: 100
`
	cfg, _, err := Parse([]byte(yaml))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Data.Synthetic.Endpoint != "http://ollama.internal:11434" {
		t.Errorf("endpoint = %q, want Ollama URL", cfg.Data.Synthetic.Endpoint)
	}
}

func TestParseUnknownFieldsProduceWarnings(t *testing.T) {
	yaml := `
apiVersion: slemify/v1
project:
  name: test-expert
  task: generation
  domain: "test"
model:
  base: mistralai/Mistral-7B-Instruct-v0.3
data:
  bucket: b
  path: p/
  synthetic:
    model: claude-sonnet
    pairs: 50
unknownField: should-warn
`
	_, warnings, err := Parse([]byte(yaml))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(warnings) == 0 {
		t.Error("expected warnings for unknown field, got none")
	}
}

func TestParseInvalidYAML(t *testing.T) {
	_, _, err := Parse([]byte(`{invalid yaml: [`))
	if err == nil {
		t.Error("expected error for invalid YAML, got nil")
	}
}
