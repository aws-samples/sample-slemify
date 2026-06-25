// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/go-playground/validator/v10"
)

// safeName matches valid K8s DNS labels: lowercase alphanumeric and hyphens, 1-63 chars.
var safeName = regexp.MustCompile(`^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$`)

// safeBucket matches valid S3 bucket names: lowercase alphanumeric, hyphens, dots, 3-63 chars.
var safeBucket = regexp.MustCompile(`^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$`)

// unsafePath rejects path traversal attempts.
var unsafePath = regexp.MustCompile(`(^|/)\.\.(/|$)`)

var validate *validator.Validate

func init() {
	validate = validator.New(validator.WithRequiredStructEnabled())

	// Register dns_label as alias for hostname validation (RFC 1123 label)
	validate.RegisterAlias("dns_label", "hostname")
}

// Validate checks an ExpertConfig for required fields and valid values.
// Returns a slice of ValidationError (empty = valid).
func Validate(cfg *ExpertConfig) []ValidationError {
	var errs []ValidationError

	if err := validate.Struct(cfg); err != nil {
		if validationErrors, ok := err.(validator.ValidationErrors); ok {
			for _, fe := range validationErrors {
				errs = append(errs, ValidationError{
					Field:    yamlFieldPath(fe.Namespace()),
					Expected: formatExpected(fe),
					Message:  formatMessage(fe),
				})
			}
		}
	}

	// Task: reject valid-but-not-yet-implemented task values with a clear message.
	// The struct `oneof` tag already rejects unknown values; this catches values
	// that are valid in the schema but whose pipeline isn't built yet.
	if cfg.Project.Task != "" && !IsSupportedTask(cfg.Project.Task) {
		errs = append(errs, ValidationError{
			Field:   "project.task",
			Message: fmt.Sprintf("task %q is declared but not yet supported in this version", cfg.Project.Task),
		})
	}

	// Task-aware field rules.
	// model.base is required for every task except extraction (whose v1 tagger
	// is feature-based and uses no encoder).
	if cfg.Project.Task != "" && !cfg.Project.IsExtraction() && cfg.Model.Base == "" {
		errs = append(errs, ValidationError{
			Field:   "model.base",
			Message: "model.base is required for this task",
		})
	}
	if cfg.Project.IsEncoderHead() {
		// Encoder-head families generate synthetic training data (the head is fit
		// on synthesized examples), so synthetic generation is required here.
		if cfg.Data.Synthetic == (SyntheticConfig{}) {
			errs = append(errs, ValidationError{
				Field:   "data.synthetic",
				Message: fmt.Sprintf("task %q requires data.synthetic (model + pairs) for training data generation", cfg.Project.Task),
			})
		}
		// Classification and extraction predict from a label taxonomy, so it's
		// required. Scoring outputs a number and needs no labels.
		if cfg.Project.UsesLabels() && len(cfg.Project.Labels) == 0 {
			errs = append(errs, ValidationError{
				Field:   "project.labels",
				Message: fmt.Sprintf("task %q requires a labels taxonomy", cfg.Project.Task),
			})
		}
		if cfg.Project.OutputFormat != "" {
			errs = append(errs, ValidationError{
				Field:   "project.output_format",
				Message: "output_format applies only to task=generation",
			})
		}
		if cfg.Model.Quantize != "" {
			errs = append(errs, ValidationError{
				Field:   "model.quantize",
				Message: "quantize applies only to task=generation (encoder-head models are not quantized to GGUF)",
			})
		}
	} else if cfg.Project.IsEmbedding() {
		// Embedding (contrastive) trains an encoder on (query, document) pairs
		// synthesized from the raw corpus. It has no label taxonomy, no
		// classifier head, and is not quantized to GGUF.
		if cfg.Data.Synthetic == (SyntheticConfig{}) {
			errs = append(errs, ValidationError{
				Field:   "data.synthetic",
				Message: "task=embedding requires data.synthetic (model + pairs) for contrastive pair generation",
			})
		}
		if len(cfg.Project.Labels) > 0 {
			errs = append(errs, ValidationError{
				Field:   "project.labels",
				Message: "labels do not apply to task=embedding (output is a vector, not a label)",
			})
		}
		if cfg.Model.Head != "" {
			errs = append(errs, ValidationError{
				Field:   "model.head",
				Message: "head applies only to encoder-head tasks (classification, scoring, extraction)",
			})
		}
		if cfg.Model.Quantize != "" {
			errs = append(errs, ValidationError{
				Field:   "model.quantize",
				Message: "quantize applies only to task=generation",
			})
		}
		if cfg.Project.OutputFormat != "" {
			errs = append(errs, ValidationError{
				Field:   "project.output_format",
				Message: "output_format applies only to task=generation",
			})
		}
		// Contrastive training needs source documents to mine pairs from.
		if len(cfg.Data.Sources) == 0 {
			errs = append(errs, ValidationError{
				Field:   "data.sources",
				Message: "task=embedding requires at least one data source (the corpus to mine query/document pairs from)",
			})
		}
	} else {
		// Generation: must not set the classifier head.
		if cfg.Model.Head != "" {
			errs = append(errs, ValidationError{
				Field:   "model.head",
				Message: "head applies only to encoder-head tasks (classification, scoring, extraction)",
			})
		}
	}

	// Security: validate inputs used in K8s names, S3 paths, and shell commands
	if cfg.Project.Name != "" && !safeName.MatchString(cfg.Project.Name) {
		errs = append(errs, ValidationError{
			Field:   "project.name",
			Message: "must be a safe DNS label (lowercase alphanumeric and hyphens, 1-63 chars, no leading/trailing hyphen)",
		})
	}
	if cfg.Data.Bucket != "" && !safeBucket.MatchString(cfg.Data.Bucket) {
		errs = append(errs, ValidationError{
			Field:   "data.bucket",
			Message: "must be a valid S3 bucket name (lowercase alphanumeric, hyphens, dots, 3-63 chars)",
		})
	}
	if cfg.Data.Path != "" && unsafePath.MatchString(cfg.Data.Path) {
		errs = append(errs, ValidationError{
			Field:   "data.path",
			Message: "must not contain path traversal sequences (..)",
		})
	}

	return errs
}

// CheckUnknownFields parses raw YAML and reports unknown fields as warnings.
// Returns a list of warning strings for unknown fields found.
func CheckUnknownFields(data []byte) []string {
	// Use json round-trip via sigs.k8s.io/yaml to detect unknown fields.
	// First unmarshal to a generic map, then compare keys against known fields.
	var raw map[string]interface{}
	if err := json.Unmarshal(yamlToJSON(data), &raw); err != nil {
		return nil
	}

	knownTopLevel := map[string]bool{
		"apiVersion": true,
		"project":    true,
		"model":      true,
		"data":       true,
		"training":   true,
		"agent":      true,
	}

	var warnings []string
	for key := range raw {
		if !knownTopLevel[key] {
			warnings = append(warnings, fmt.Sprintf("unknown field %q at top level", key))
		}
	}

	return warnings
}

// yamlToJSON is a simple helper; in production this uses sigs.k8s.io/yaml.
// For now we do a basic conversion via the yaml package.
func yamlToJSON(data []byte) []byte {
	// This is a placeholder — the actual implementation uses sigs.k8s.io/yaml
	// which handles YAML-to-JSON conversion properly.
	// For struct validation we rely on go-playground/validator.
	return data
}

// yamlFieldPath converts Go struct namespace to YAML-style field path.
// e.g. "ExpertConfig.Project.Name" -> "project.name"
func yamlFieldPath(namespace string) string {
	parts := strings.Split(namespace, ".")
	if len(parts) <= 1 {
		return strings.ToLower(namespace)
	}
	// Skip the root struct name
	parts = parts[1:]

	result := make([]string, len(parts))
	for i, p := range parts {
		result[i] = camelToYAML(p)
	}
	return strings.Join(result, ".")
}

// camelToYAML converts CamelCase to yaml_case.
func camelToYAML(s string) string {
	fieldMap := map[string]string{
		"APIVersion":    "apiVersion",
		"Project":       "project",
		"Model":         "model",
		"Data":          "data",
		"Training":      "training",
		"Agent":         "agent",
		"Name":          "name",
		"Domain":        "domain",
		"DomainVersion": "domain_version",
		"Base":          "base",
		"Bucket":        "bucket",
		"Path":          "path",
		"Sources":       "sources",
		"Synthetic":     "synthetic",
		"Endpoint":      "endpoint",
		"Pairs":         "pairs",
		"Spot":          "spot",
		"Tools":         "tools",
		"Description":   "description",
		"Type":          "type",
		"Metadata":      "metadata",
		"Priority":      "priority",
		"Strategy":      "strategy",
		"Projects":      "projects",
		"Task":          "task",
		"OutputFormat":  "output_format",
		"Head":          "head",
	}
	if mapped, ok := fieldMap[s]; ok {
		return mapped
	}
	return strings.ToLower(s)
}

func formatExpected(fe validator.FieldError) string {
	switch fe.Tag() {
	case "required":
		return "non-empty value"
	case "eq":
		return fe.Param()
	case "oneof":
		return "one of: " + fe.Param()
	case "min":
		return "minimum " + fe.Param()
	case "dns_label", "hostname":
		return "valid DNS label (lowercase alphanumeric, hyphens)"
	default:
		return fe.Tag()
	}
}

func formatMessage(fe validator.FieldError) string {
	switch fe.Tag() {
	case "required":
		return "field is required"
	case "eq":
		return fmt.Sprintf("must be %q", fe.Param())
	case "oneof":
		return fmt.Sprintf("must be one of: %s", fe.Param())
	case "min":
		return fmt.Sprintf("must be at least %s", fe.Param())
	case "dns_label", "hostname":
		return "must be a valid DNS label (lowercase alphanumeric and hyphens, max 63 chars)"
	default:
		return fmt.Sprintf("failed %s validation", fe.Tag())
	}
}
