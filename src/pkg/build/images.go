// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package build

import "fmt"

// DefaultRegistry is the default public container image registry.
const DefaultRegistry = "ghcr.io/slemify"

// ImageRef returns the full image reference for a slemify container.
// If registry is empty, uses the default public registry.
func ImageRef(registry, name, tag string) string {
	if registry == "" {
		registry = DefaultRegistry
	}
	return fmt.Sprintf("%s/%s:%s", registry, name, tag)
}

// DataPipelineImage returns the data-pipeline container image reference.
func DataPipelineImage(registry string) string {
	return ImageRef(registry, "data-pipeline", "latest")
}

// TrainingImage returns the training container image reference.
// Note: Training now uses the official unsloth/unsloth image directly.
// This function is retained for backward compatibility.
func TrainingImage(registry string) string {
	return ImageRef(registry, "training", "latest")
}
