// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

import (
	"strings"
	"testing"
)

func validConfig() *ExpertConfig {
	return &ExpertConfig{
		APIVersion: "slemify/v1",
		Project: ProjectConfig{
			Name:   "karpenter-expert",
			Domain: "Karpenter configuration",
			Task:   TaskGeneration,
		},
		Model: ModelConfig{Base: "mistralai/Mistral-7B-Instruct-v0.3"},
		Data: DataConfig{
			Bucket: "my-bucket",
			Path:   "data/",
			Synthetic: SyntheticConfig{
				Model: "claude-sonnet",
				Pairs: 500,
			},
		},
		Training: TrainingConfig{Spot: true},
	}
}

func TestValidateValidConfig(t *testing.T) {
	errs := Validate(validConfig())
	if len(errs) > 0 {
		t.Errorf("expected no errors, got %d: %v", len(errs), errs)
	}
}

func TestValidateMissingAPIVersion(t *testing.T) {
	cfg := validConfig()
	cfg.APIVersion = ""
	errs := Validate(cfg)
	if !hasFieldError(errs, "apiVersion") {
		t.Errorf("expected error on apiVersion, got: %v", errs)
	}
}

func TestValidateWrongAPIVersion(t *testing.T) {
	cfg := validConfig()
	cfg.APIVersion = "slemify/v2"
	errs := Validate(cfg)
	if !hasFieldError(errs, "apiVersion") {
		t.Errorf("expected error on apiVersion for wrong version, got: %v", errs)
	}
}

func TestValidateMissingProjectName(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Name = ""
	errs := Validate(cfg)
	if !hasFieldError(errs, "project.name") {
		t.Errorf("expected error on project.name, got: %v", errs)
	}
}

func TestValidateInvalidProjectName(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Name = "INVALID_NAME!"
	errs := Validate(cfg)
	if !hasFieldError(errs, "project.name") {
		t.Errorf("expected error on project.name for invalid DNS label, got: %v", errs)
	}
}

func TestValidateMissingDomain(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Domain = ""
	errs := Validate(cfg)
	if !hasFieldError(errs, "project.domain") {
		t.Errorf("expected error on project.domain, got: %v", errs)
	}
}

func TestValidateMissingTask(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Task = ""
	errs := Validate(cfg)
	if !hasFieldError(errs, "project.task") {
		t.Errorf("expected error on project.task when empty, got: %v", errs)
	}
}

func TestValidateUnknownTask(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Task = "translation"
	errs := Validate(cfg)
	if !hasFieldError(errs, "project.task") {
		t.Errorf("expected error on unknown task value, got: %v", errs)
	}
}

func TestValidateExtractionValid(t *testing.T) {
	// extraction is supported (encoder-head family) and predicts typed spans
	// from a label taxonomy (the entity types), so labels are required.
	cfg := validConfig()
	cfg.Project.Task = TaskExtraction
	cfg.Model.Base = "" // extraction (feature tagger) needs no encoder
	cfg.Model.Quantize = "" // quantize not allowed for encoder-head
	cfg.Project.Labels = map[string][]string{"entities": {"SERVICE", "ERROR"}}
	errs := Validate(cfg)
	if len(errs) > 0 {
		t.Errorf("expected valid extraction config, got: %v", errs)
	}
}

func TestValidateExtractionRequiresLabels(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Task = TaskExtraction
	cfg.Model.Base = "BAAI/bge-base-en-v1.5"
	cfg.Model.Quantize = ""
	// no labels set
	errs := Validate(cfg)
	if !hasFieldError(errs, "project.labels") {
		t.Errorf("expected labels-required error for extraction, got: %v", errs)
	}
}

func TestValidateScoringValid(t *testing.T) {
	// scoring is supported, needs no labels, and outputs a number.
	cfg := validConfig()
	cfg.Project.Task = TaskScoring
	cfg.Model.Base = "BAAI/bge-base-en-v1.5"
	cfg.Model.Quantize = "" // quantize not allowed for encoder-head
	cfg.Project.Labels = nil
	errs := Validate(cfg)
	if len(errs) > 0 {
		t.Errorf("expected valid scoring config (no labels required), got: %v", errs)
	}
}

func TestValidateEmbeddingValid(t *testing.T) {
	// embedding is supported, needs sources (a corpus), no labels, no head.
	cfg := validConfig()
	cfg.Project.Task = TaskEmbedding
	cfg.Model.Base = "BAAI/bge-base-en-v1.5"
	cfg.Model.Quantize = ""
	cfg.Model.Head = ""
	cfg.Project.Labels = nil
	cfg.Data.Sources = []SourceConfig{{Path: "queries/", Type: "raw"}}
	errs := Validate(cfg)
	if len(errs) > 0 {
		t.Errorf("expected valid embedding config, got: %v", errs)
	}
}

func TestValidateEmbeddingRequiresSources(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Task = TaskEmbedding
	cfg.Model.Base = "BAAI/bge-base-en-v1.5"
	cfg.Model.Quantize = ""
	cfg.Project.Labels = nil
	cfg.Data.Sources = nil
	errs := Validate(cfg)
	if !hasFieldError(errs, "data.sources") {
		t.Errorf("expected sources-required error for embedding, got: %v", errs)
	}
}

func TestValidateEmbeddingRejectsLabels(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Task = TaskEmbedding
	cfg.Model.Base = "BAAI/bge-base-en-v1.5"
	cfg.Model.Quantize = ""
	cfg.Project.Labels = map[string][]string{"routing": {"a", "b"}}
	cfg.Data.Sources = []SourceConfig{{Path: "queries/", Type: "raw"}}
	errs := Validate(cfg)
	if !hasFieldError(errs, "project.labels") {
		t.Errorf("expected labels-rejected error for embedding, got: %v", errs)
	}
}

func TestValidateRerankingNotSupported(t *testing.T) {
	// reranking is a valid schema value but intentionally not a Slemify task
	// (a strong cross-encoder shouldn't be fine-tuned on synthetic data; it's
	// shown as a CPU-serving pattern in the demo instead).
	cfg := validConfig()
	cfg.Project.Task = TaskReranking
	errs := Validate(cfg)
	if !hasFieldError(errs, "project.task") {
		t.Errorf("expected 'not yet supported' error for reranking, got: %v", errs)
	}
}

func TestValidateClassificationRequiresLabels(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Task = TaskClassification
	cfg.Model.Base = "BAAI/bge-base-en-v1.5"
	cfg.Model.Quantize = "" // quantize not allowed for classification
	// no labels set
	errs := Validate(cfg)
	if !hasFieldError(errs, "project.labels") {
		t.Errorf("expected labels-required error for classification, got: %v", errs)
	}
}

func TestValidateClassificationValid(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Task = TaskClassification
	cfg.Project.Labels = map[string][]string{"routing": {"a", "b"}}
	cfg.Model.Base = "BAAI/bge-base-en-v1.5"
	cfg.Model.Quantize = ""
	errs := Validate(cfg)
	if len(errs) > 0 {
		t.Errorf("expected valid classification config, got: %v", errs)
	}
}

func TestValidateClassificationRejectsQuantize(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Task = TaskClassification
	cfg.Project.Labels = map[string][]string{"routing": {"a", "b"}}
	cfg.Model.Base = "BAAI/bge-base-en-v1.5"
	cfg.Model.Quantize = "q4_k_m"
	errs := Validate(cfg)
	if !hasFieldError(errs, "model.quantize") {
		t.Errorf("expected quantize-not-allowed error for classification, got: %v", errs)
	}
}

func TestValidateGenerationTaskSupported(t *testing.T) {
	cfg := validConfig()
	cfg.Project.Task = TaskGeneration
	errs := Validate(cfg)
	if hasFieldError(errs, "project.task") {
		t.Errorf("generation task should be supported, got error: %v", errs)
	}
}

func TestValidateMissingModelBase(t *testing.T) {
	cfg := validConfig()
	cfg.Model.Base = ""
	errs := Validate(cfg)
	if len(errs) == 0 {
		t.Fatal("expected validation error for empty model.base")
	}
	// The error should reference either "model.base" or "model"
	found := false
	for _, e := range errs {
		if strings.Contains(e.Field, "model") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected error referencing model, got: %v", errs)
	}
}

func TestValidateMissingBucket(t *testing.T) {
	cfg := validConfig()
	cfg.Data.Bucket = ""
	errs := Validate(cfg)
	if !hasFieldError(errs, "data.bucket") {
		t.Errorf("expected error on data.bucket, got: %v", errs)
	}
}

func TestValidateSyntheticPairsTooLow(t *testing.T) {
	cfg := validConfig()
	cfg.Data.Synthetic.Pairs = 5
	errs := Validate(cfg)
	if !hasFieldError(errs, "data.synthetic.pairs") {
		t.Errorf("expected error on synthetic.pairs < 10, got: %v", errs)
	}
}

func TestValidateSyntheticPairsMinimum(t *testing.T) {
	cfg := validConfig()
	cfg.Data.Synthetic.Pairs = 10
	errs := Validate(cfg)
	if hasFieldError(errs, "data.synthetic.pairs") {
		t.Errorf("pairs=10 should be valid, got error: %v", errs)
	}
}

func TestValidateInvalidSourceType(t *testing.T) {
	cfg := validConfig()
	cfg.Data.Sources = []SourceConfig{
		{Path: "data/", Type: "invalid-type"},
	}
	errs := Validate(cfg)
	if !hasFieldError(errs, "type") {
		t.Errorf("expected error on invalid source type, got: %v", errs)
	}
}

func TestValidateValidSourceTypes(t *testing.T) {
	validTypes := []string{"github-issues", "github-discussions", "documentation", "source-code", "documents", "raw"}
	for _, st := range validTypes {
		cfg := validConfig()
		cfg.Data.Sources = []SourceConfig{
			{Path: "data/", Type: st},
		}
		errs := Validate(cfg)
		if hasFieldError(errs, "type") {
			t.Errorf("source type %q should be valid, got error: %v", st, errs)
		}
	}
}

func TestValidateMultipleErrors(t *testing.T) {
	cfg := &ExpertConfig{} // everything missing
	errs := Validate(cfg)
	if len(errs) < 3 {
		t.Errorf("expected multiple errors for empty config, got %d: %v", len(errs), errs)
	}
}

func TestValidationErrorFormat(t *testing.T) {
	cfg := validConfig()
	cfg.APIVersion = ""
	errs := Validate(cfg)
	if len(errs) == 0 {
		t.Fatal("expected errors")
	}
	// Check that Error() method works
	errStr := errs[0].Error()
	if errStr == "" {
		t.Error("Error() returned empty string")
	}
}

// hasFieldError checks if any ValidationError references the given field substring.
func hasFieldError(errs []ValidationError, field string) bool {
	for _, e := range errs {
		if strings.Contains(e.Field, field) {
			return true
		}
	}
	return false
}
