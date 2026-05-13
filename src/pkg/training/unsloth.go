// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package training

import (
	"bytes"
	"embed"
	"fmt"
	"text/template"

	"github.com/aws-samples/sample-slemify/pkg/config"
)

//go:embed train.py.tmpl
var trainTemplate embed.FS

var tmpl = template.Must(template.ParseFS(trainTemplate, "train.py.tmpl"))

type trainParams struct {
	BaseModel          string
	Epochs             int
	LearningRate       float64
	WarmupRatio        float64
	Scheduler          string
	CheckpointInterval int
	Resume             string
	Quantize           string
}

// UnslothTrainingScript generates a Python script for Unsloth-based QLoRA fine-tuning.
func UnslothTrainingScript(cfg *config.ExpertConfig, sized config.SizedConfig) string {
	resume := "False"
	if cfg.Training.Incremental {
		resume = "True"
	}

	params := trainParams{
		BaseModel:          cfg.Model.Base,
		Epochs:             sized.Epochs,
		LearningRate:       sized.LearningRate,
		WarmupRatio:        sized.WarmupRatio,
		Scheduler:          sized.Scheduler,
		CheckpointInterval: sized.CheckpointInterval,
		Resume:             resume,
		Quantize:           cfg.Model.QuantizeType(),
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, params); err != nil {
		panic(fmt.Sprintf("executing training template: %v", err))
	}
	return buf.String()
}
