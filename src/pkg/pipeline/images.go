// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package pipeline

import (
	"fmt"
	"strings"
)

// PipelineContext holds shared configuration for all pipeline stages.
// Passed into stage constructors instead of relying on package-level globals.
type PipelineContext struct {
	Registry       string // Container image registry prefix
	ServiceAccount string // K8s service account name for Pod Identity
	NoWait         bool   // Submit Jobs without waiting for completion
	UseS3Mount     bool   // Mount model from S3 via CSI driver (vs init container download)
}

// NewPipelineContext creates a PipelineContext with default values.
func NewPipelineContext() *PipelineContext {
	return &PipelineContext{
		Registry: "ghcr.io/slemify",
	}
}

// Image returns the full image reference for a slemify container.
func (pc *PipelineContext) Image(name string) string {
	return fmt.Sprintf("%s/%s:latest", pc.Registry, name)
}

// SplitYAMLDocs splits a multi-document YAML string into individual documents.
func SplitYAMLDocs(yaml string) []string {
	var docs []string
	for _, doc := range strings.Split(yaml, "---") {
		doc = strings.TrimSpace(doc)
		if doc != "" {
			docs = append(docs, doc)
		}
	}
	return docs
}
