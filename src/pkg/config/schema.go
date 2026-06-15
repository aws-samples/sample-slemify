// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

// ExpertConfig is the top-level schema for a single expert.
type ExpertConfig struct {
	APIVersion string           `json:"apiVersion" yaml:"apiVersion" validate:"required,eq=slemify/v1"`
	Project    ProjectConfig    `json:"project" yaml:"project" validate:"required"`
	Model      ModelConfig      `json:"model" yaml:"model"`
	Data       DataConfig       `json:"data" yaml:"data" validate:"required"`
	Training   TrainingConfig   `json:"training" yaml:"training"`
	Evaluation EvaluationConfig `json:"evaluation,omitempty" yaml:"evaluation,omitempty"`
}

// Task values. Each maps onto one of three implementation families:
//   - generation:     causal LM, GPU fine-tune, CPU serve (llama.cpp/GGUF)
//   - classification, scoring, extraction: frozen encoder + trained head,
//     CPU train, CPU serve (encoder-head family)
//   - embedding:       contrastive encoder, CPU serve (embedding family)
//
// reranking is a valid schema value but is intentionally NOT a Slemify task:
// fine-tuning a strong cross-encoder on synthetic data degrades it (it needs
// curated relevance judgments we can't synthesize), and serving a stock
// cross-encoder is just a CPU-serving pattern — shown in the k8s-autoscaling
// demo — not a model Slemify builds. See the README FAQ.
const (
	TaskGeneration     = "generation"
	TaskClassification = "classification"
	TaskScoring        = "scoring"
	TaskExtraction     = "extraction"
	TaskReranking      = "reranking"
	TaskEmbedding      = "embedding"
)

// supportedTasks lists task values whose full pipeline is implemented in this
// version. Other valid task values are accepted by the schema but rejected by
// validation with a clear "not yet supported" message.
var supportedTasks = map[string]bool{
	TaskGeneration:     true,
	TaskClassification: true,
	TaskScoring:        true,
	TaskExtraction:     true,
	TaskEmbedding:      true,
}

// IsSupportedTask returns true if the task's pipeline is implemented.
func IsSupportedTask(task string) bool {
	return supportedTasks[task]
}

type ProjectConfig struct {
	Name          string              `json:"name" yaml:"name" validate:"required,dns_label"`
	Domain        string              `json:"domain" yaml:"domain" validate:"required"`
	DomainVersion string              `json:"domain_version,omitempty" yaml:"domain_version,omitempty"`
	Labels        map[string][]string `json:"labels,omitempty" yaml:"labels,omitempty"`
	// Task selects the model family and pipeline. Required, no default.
	Task string `json:"task" yaml:"task" validate:"required,oneof=generation classification scoring extraction reranking embedding"`
	// OutputFormat applies only to task=generation. Currently only free_form is
	// used (the auditor); pipe_delimited was retired in Phase 2 when classification
	// tasks moved to the encoder-head path.
	OutputFormat string `json:"output_format,omitempty" yaml:"output_format,omitempty" validate:"omitempty,oneof=free_form"`
}

// TaskType returns the configured task value.
func (p ProjectConfig) TaskType() string {
	return p.Task
}

// IsGeneration returns true for the generative (causal LM) family.
func (p ProjectConfig) IsGeneration() bool {
	return p.Task == TaskGeneration
}

// IsEncoderHead returns true for the frozen-encoder + trained-head family
// (classification, scoring, extraction).
func (p ProjectConfig) IsEncoderHead() bool {
	switch p.Task {
	case TaskClassification, TaskScoring, TaskExtraction:
		return true
	}
	return false
}

// IsEmbedding returns true for the contrastive embedding family.
func (p ProjectConfig) IsEmbedding() bool {
	return p.Task == TaskEmbedding
}

// IsScoring returns true for the regression/scoring task (numeric output).
func (p ProjectConfig) IsScoring() bool {
	return p.Task == TaskScoring
}

// IsExtraction returns true for the token-level entity extraction task
// (output is a list of typed spans). v1 serves a feature-based token tagger on
// CPU rather than the frozen encoder, so it skips the ONNX encoder artifact.
func (p ProjectConfig) IsExtraction() bool {
	return p.Task == TaskExtraction
}

// UsesLabels returns true for tasks whose output is drawn from a label
// taxonomy (classification, extraction). Scoring outputs a number, embedding
// outputs a vector, and generation is free-form — none of those need labels.
func (p ProjectConfig) UsesLabels() bool {
	switch p.Task {
	case TaskClassification, TaskExtraction:
		return true
	}
	return false
}

// IsFreeForm returns true if a generation expert uses free-form output
// (reasoning/audit traces). free_form is the only output_format value;
// pipe_delimited was retired in Phase 2 (classification moved to the
// encoder-head path).
func (p ProjectConfig) IsFreeForm() bool {
	return p.IsGeneration() && p.OutputFormat == "free_form"
}

type ModelConfig struct {
	// Base is the model identifier: a causal LM for generation, an encoder for
	// classification/scoring/embedding. Not required for extraction, whose v1
	// tagger is feature-based and uses no encoder. Per-task requiredness is
	// enforced in the validator.
	Base     string `json:"base" yaml:"base" validate:"omitempty"`
	Quantize string `json:"quantize,omitempty" yaml:"quantize,omitempty" validate:"omitempty,oneof=q4_k_m q8_0 f16 none"` // default: q4_k_m (generation only)
	// Head selects the classifier head for encoder-head tasks. Ignored for
	// generation/embedding.
	Head string `json:"head,omitempty" yaml:"head,omitempty" validate:"omitempty,oneof=logistic linear mlp"`
}

// HeadType returns the effective classifier head, defaulting to logistic.
// Only meaningful for encoder-head tasks.
func (m ModelConfig) HeadType() string {
	if m.Head == "" {
		return "logistic"
	}
	return m.Head
}

// QuantizeType returns the effective quantization type, defaulting to q4_k_m.
func (m ModelConfig) QuantizeType() string {
	if m.Quantize == "" {
		return "q4_k_m"
	}
	return m.Quantize
}

// GGUFFilename returns the GGUF model filename based on quantization type.
func (m ModelConfig) GGUFFilename() string {
	qt := m.QuantizeType()
	if qt == "none" || qt == "f16" {
		return "model-f16.gguf"
	}
	return "model-" + qt + ".gguf"
}

// QuantizeLabel returns a human-readable label for the quantization type.
func (m ModelConfig) QuantizeLabel() string {
	switch m.QuantizeType() {
	case "none", "f16":
		return "F16 (no quantization)"
	case "q8_0":
		return "Q8_0"
	default:
		return "Q4_K_M"
	}
}

type DataConfig struct {
	Bucket     string           `json:"bucket" yaml:"bucket" validate:"required"`
	Path       string           `json:"path" yaml:"path" validate:"required"`
	Sources    []SourceConfig   `json:"sources,omitempty" yaml:"sources,omitempty" validate:"omitempty,dive"`
	Synthetic  SyntheticConfig  `json:"synthetic" yaml:"synthetic" validate:"required"`
	Evaluation *EvalDataConfig  `json:"evaluation,omitempty" yaml:"evaluation,omitempty"`
}

type SourceConfig struct {
	Path     string          `json:"path" yaml:"path" validate:"required"`
	Type     string          `json:"type,omitempty" yaml:"type,omitempty" validate:"omitempty,oneof=github-issues github-discussions documentation source-code documents raw"`
	Metadata *SourceMetadata `json:"metadata,omitempty" yaml:"metadata,omitempty"`
}

type SourceMetadata struct {
	Priority string `json:"priority,omitempty" yaml:"priority,omitempty" validate:"omitempty,oneof=high medium low"`
}

// SyntheticConfig controls synthetic training data generation.
// If Endpoint is empty, Bedrock is assumed. If set, any OpenAI-compatible API is used.
type SyntheticConfig struct {
	Model    string `json:"model" yaml:"model" validate:"required"`
	Endpoint string `json:"endpoint,omitempty" yaml:"endpoint,omitempty"`
	Pairs    int    `json:"pairs" yaml:"pairs" validate:"required,min=10"`
}

// EvalDataConfig controls independent evaluation data generation.
// Uses a separate model and/or separate source data to produce eval pairs
// that are independent from the training set.
type EvalDataConfig struct {
	Model   string         `json:"model" yaml:"model" validate:"required"`
	Pairs   int            `json:"pairs" yaml:"pairs" validate:"required,min=10"`
	Sources []SourceConfig `json:"sources,omitempty" yaml:"sources,omitempty"`
}

type TrainingConfig struct {
	Spot        bool `json:"spot" yaml:"spot"`
	Epochs      int  `json:"epochs,omitempty" yaml:"epochs,omitempty" validate:"omitempty,min=1,max=20"`      // override auto-sized epochs
	Incremental bool `json:"incremental,omitempty" yaml:"incremental,omitempty"` // resume from last checkpoint with fewer epochs
}

// EvaluationConfig holds optional test prompts for inference benchmarking.
// If Prompts is empty, the benchmark samples from eval.jsonl automatically.
// ExpectedAnswers is optional — if provided, responses are scored against them.
type EvaluationConfig struct {
	Prompts         []string `json:"prompts,omitempty" yaml:"prompts,omitempty"`
	ExpectedAnswers []string `json:"expected_answers,omitempty" yaml:"expected_answers,omitempty"`
}

// SizedConfig holds the auto-computed infrastructure values produced by AutoSize.
type SizedConfig struct {
	TrainingGPU        string
	TrainingInstance   string // display only — Karpenter selects actual instance
	InferenceInstance  string // display only — Karpenter selects actual instance
	InferenceCPU       string // CPU request for inference pod (e.g., "4")
	InferenceMemory    string // Memory request for inference pod (e.g., "8Gi")
	InferenceThreads   string // llama.cpp --threads flag
	CheckpointInterval int    // steps
	Epochs             int
	LearningRate       float64
	WarmupRatio        float64
	Scheduler          string
	EarlyStopPatience  int
	KEDAMaxReplicas    int
	MaxOutputTokens    int // from output_stats.json (p95 + 20% headroom), 0 = use defaults
	ReasoningBudget    int // avg_output_tokens / 2 for free-form, 0 for classification
}

// ValidationError represents a single config validation failure.
type ValidationError struct {
	Field    string `json:"field"`
	Expected string `json:"expected,omitempty"`
	Message  string `json:"message"`
}

// Error implements the error interface.
func (e ValidationError) Error() string {
	if e.Field != "" {
		return e.Field + ": " + e.Message
	}
	return e.Message
}
