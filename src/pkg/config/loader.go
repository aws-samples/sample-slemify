// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

import (
	"fmt"
	"os"

	"sigs.k8s.io/yaml"
)

// Load reads and parses an Expert Config YAML file.
// It uses sigs.k8s.io/yaml for Kubernetes-compatible YAML parsing.
func Load(path string) (*ExpertConfig, []string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("reading %s: %w", path, err)
	}

	return Parse(data)
}

// Parse parses raw YAML bytes into an ExpertConfig.
// Returns the config, any unknown field warnings, and an error if parsing fails.
func Parse(data []byte) (*ExpertConfig, []string, error) {
	var cfg ExpertConfig

	// Strict unmarshal: rejects unknown fields
	if err := yaml.UnmarshalStrict(data, &cfg); err != nil {
		// Try lenient parse to detect which fields are unknown
		warnings := detectUnknownFields(data)
		if len(warnings) > 0 {
			// Parse leniently to still return a usable config
			if err2 := yaml.Unmarshal(data, &cfg); err2 != nil {
				return nil, nil, fmt.Errorf("parsing config: %w", err2)
			}
			cfg.ApplyDefaults()
			return &cfg, warnings, nil
		}
		return nil, nil, fmt.Errorf("parsing config: %w", err)
	}

	cfg.ApplyDefaults()
	return &cfg, nil, nil
}

// Default base models per task family, used when model.base is empty. The
// encoder default matches what the trainer container falls back to; the
// generation default is the small-MoE the reference example was measured with.
const (
	DefaultEncoderBase    = "BAAI/bge-base-en-v1.5"
	DefaultGenerationBase = "Qwen/Qwen3-30B-A3B-Instruct-2507"
)

// ApplyDefaults fills fields that have a sensible task-dependent default so a
// config can leave them empty. Only model.base today.
func (c *ExpertConfig) ApplyDefaults() {
	if c.Model.Base != "" {
		return
	}
	switch {
	case c.Project.IsGeneration():
		c.Model.Base = DefaultGenerationBase
	case c.Project.IsExtraction():
		// feature-based tagger, no encoder
	case c.Project.Task != "":
		c.Model.Base = DefaultEncoderBase
	}
}

// detectUnknownFields attempts to identify unknown fields by comparing
// strict vs lenient parsing results.
func detectUnknownFields(data []byte) []string {
	var strict ExpertConfig
	strictErr := yaml.UnmarshalStrict(data, &strict)
	if strictErr == nil {
		return nil
	}

	var lenient ExpertConfig
	if err := yaml.Unmarshal(data, &lenient); err != nil {
		return nil
	}

	// If lenient succeeds but strict fails, there are unknown fields
	return []string{fmt.Sprintf("config contains unknown fields: %v", strictErr)}
}
