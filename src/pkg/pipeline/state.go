// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package pipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// State represents the full pipeline state persisted to S3.
type State struct {
	Project      string                 `json:"project"`
	CurrentStage Stage                  `json:"current_stage"`
	Stages       map[Stage]StageResult  `json:"stages"`
}

// NewState creates a fresh pipeline state for a project.
func NewState(project string) *State {
	stages := make(map[Stage]StageResult)
	for _, s := range StageOrder {
		stages[s] = StageResult{
			Stage:  s,
			Status: StatusPending,
		}
	}
	return &State{
		Project: project,
		Stages:  stages,
	}
}

// StateStore handles reading/writing pipeline state to S3.
type StateStore struct {
	client *s3.Client
	bucket string
}

// NewStateStore creates an S3-backed state store.
func NewStateStore(ctx context.Context, bucket string) (*StateStore, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("loading AWS config: %w", err)
	}

	return &StateStore{
		client: s3.NewFromConfig(cfg),
		bucket: bucket,
	}, nil
}

// stateKey returns the S3 key for a project's pipeline state.
func stateKey(project string) string {
	return fmt.Sprintf("%s/state/pipeline-state.json", project)
}

// Load reads pipeline state from S3. Returns a fresh state if not found.
func (ss *StateStore) Load(ctx context.Context, project string) (*State, error) {
	result, err := ss.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(ss.bucket),
		Key:    aws.String(stateKey(project)),
	})
	if err != nil {
		// If not found, return fresh state
		return NewState(project), nil
	}
	defer result.Body.Close()

	data, err := io.ReadAll(result.Body)
	if err != nil {
		return nil, fmt.Errorf("reading state from S3: %w", err)
	}

	var state State
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("parsing pipeline state: %w", err)
	}

	return &state, nil
}

// Save writes pipeline state to S3.
func (ss *StateStore) Save(ctx context.Context, state *State) error {
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling pipeline state: %w", err)
	}

	_, err = ss.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(ss.bucket),
		Key:         aws.String(stateKey(state.Project)),
		Body:        bytes.NewReader(data),
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		return fmt.Errorf("writing state to S3: %w", err)
	}

	return nil
}
