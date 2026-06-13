// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package report

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
)

// EncoderHeadMetrics mirrors the metrics.json written by the classifier
// training job (encoder-head tasks). It holds exact-match accuracy and
// per-class precision/recall/F1.
type EncoderHeadMetrics struct {
	EmbeddingDim int                    `json:"embedding_dim"`
	Head         string                 `json:"head"`
	NumClasses   int                    `json:"num_classes"`
	TrainSamples int                    `json:"train_samples"`
	EvalSamples  int                    `json:"eval_samples"`
	Accuracy     float64                `json:"accuracy"`
	Correct      int                    `json:"correct"`
	Total        int                    `json:"total"`
	PerClass     map[string]ClassMetric `json:"per_class"`
	EmbedMSPerQ  float64                `json:"embed_ms_per_query"`
}

// ClassMetric holds per-class precision/recall/F1.
type ClassMetric struct {
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
}

// ScoringMetrics mirrors the metrics.json written by the classifier training
// job for task=scoring (regression head). Holds error/correlation metrics.
type ScoringMetrics struct {
	Task         string  `json:"task"`
	EmbeddingDim int     `json:"embedding_dim"`
	Head         string  `json:"head"`
	TrainSamples int     `json:"train_samples"`
	EvalSamples  int     `json:"eval_samples"`
	MAE          float64 `json:"mae"`
	RMSE         float64 `json:"rmse"`
	R2           float64 `json:"r2"`
	Correlation  float64 `json:"correlation"`
	Total        int     `json:"total"`
	EmbedMSPerQ  float64 `json:"embed_ms_per_query"`
}

// S3Downloader abstracts the single S3 read needed here, avoiding a dependency
// on the k8s package (and any import cycle).
type S3Downloader interface {
	DownloadFromS3(ctx context.Context, bucket, key string) (string, error)
}

// LoadClassificationMetrics reads models/<project>/metrics.json from S3.
func LoadClassificationMetrics(ctx context.Context, dl S3Downloader, bucket, project string) (*EncoderHeadMetrics, error) {
	key := fmt.Sprintf("models/%s/metrics.json", project)
	data, err := dl.DownloadFromS3(ctx, bucket, key)
	if err != nil {
		return nil, fmt.Errorf("downloading metrics.json: %w", err)
	}
	var m EncoderHeadMetrics
	if err := json.Unmarshal([]byte(data), &m); err != nil {
		return nil, fmt.Errorf("parsing metrics.json: %w", err)
	}
	return &m, nil
}

// PrintEncoderHeadMetrics renders the classification (encoder-head) report.
func PrintEncoderHeadMetrics(m *EncoderHeadMetrics) {
	fmt.Printf("\n  ━━━ Classification Report (encoder-head) ━━━\n")
	fmt.Printf("  Accuracy:    %.1f%% (%d/%d) exact-match\n", m.Accuracy*100, m.Correct, m.Total)
	fmt.Printf("  Head:        %s (%d classes, %dd embeddings)\n", m.Head, m.NumClasses, m.EmbeddingDim)
	fmt.Printf("  Data:        %d train / %d eval\n", m.TrainSamples, m.EvalSamples)
	if m.EmbedMSPerQ > 0 {
		fmt.Printf("  Latency:     ~%.0fms/query (embed) + <1ms classify (CPU)\n", m.EmbedMSPerQ)
	}

	// Show weakest classes by F1 (ascending), to highlight where to improve data.
	type ce struct {
		name string
		m    ClassMetric
	}
	var classes []ce
	for name, cm := range m.PerClass {
		classes = append(classes, ce{name, cm})
	}
	sort.Slice(classes, func(i, j int) bool { return classes[i].m.F1 < classes[j].m.F1 })

	if len(classes) > 0 {
		fmt.Printf("  Per-class (weakest first):\n")
		for i, c := range classes {
			if i >= 5 {
				break
			}
			fmt.Printf("    %-28s P=%.2f R=%.2f F1=%.2f\n", c.name, c.m.Precision, c.m.Recall, c.m.F1)
		}
	}
	fmt.Printf("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
}

// LoadScoringMetrics reads models/<project>/metrics.json from S3 for a
// scoring (regression) head.
func LoadScoringMetrics(ctx context.Context, dl S3Downloader, bucket, project string) (*ScoringMetrics, error) {
	key := fmt.Sprintf("models/%s/metrics.json", project)
	data, err := dl.DownloadFromS3(ctx, bucket, key)
	if err != nil {
		return nil, fmt.Errorf("downloading metrics.json: %w", err)
	}
	var m ScoringMetrics
	if err := json.Unmarshal([]byte(data), &m); err != nil {
		return nil, fmt.Errorf("parsing metrics.json: %w", err)
	}
	return &m, nil
}

// PrintScoringMetrics renders the scoring (encoder-head regression) report.
func PrintScoringMetrics(m *ScoringMetrics) {
	fmt.Printf("\n  ━━━ Scoring Report (encoder-head regression) ━━━\n")
	fmt.Printf("  MAE:         %.4f (mean absolute error, lower is better)\n", m.MAE)
	fmt.Printf("  RMSE:        %.4f\n", m.RMSE)
	fmt.Printf("  R²:          %.3f (1.0 = perfect, 0 = predicts the mean)\n", m.R2)
	fmt.Printf("  Correlation: %.3f (predicted vs true score)\n", m.Correlation)
	fmt.Printf("  Head:        %s (%dd embeddings)\n", m.Head, m.EmbeddingDim)
	fmt.Printf("  Data:        %d train / %d eval\n", m.TrainSamples, m.EvalSamples)
	if m.EmbedMSPerQ > 0 {
		fmt.Printf("  Latency:     ~%.0fms/query (embed) + <1ms score (CPU)\n", m.EmbedMSPerQ)
	}
	fmt.Printf("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
}
