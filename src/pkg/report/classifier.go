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

// RetrievalMetrics holds recall@k and MRR for a retrieval run.
type RetrievalMetrics struct {
	Recall1     float64 `json:"recall@1"`
	Recall5     float64 `json:"recall@5"`
	Recall10    float64 `json:"recall@10"`
	MRR         float64 `json:"mrr"`
	EvalQueries int     `json:"eval_queries"`
}

// EmbeddingMetrics mirrors the metrics.json written by the embedding training
// job (contrastive fine-tune). It holds baseline (stock encoder) and tuned
// retrieval metrics so the gain is visible.
type EmbeddingMetrics struct {
	Task         string           `json:"task"`
	EmbeddingDim int              `json:"embedding_dim"`
	TrainSamples int              `json:"train_samples"`
	EvalQueries  int              `json:"eval_queries"`
	CorpusSize   int              `json:"corpus_size"`
	Epochs       int              `json:"epochs"`
	TrainSeconds float64          `json:"train_seconds"`
	Baseline     RetrievalMetrics `json:"baseline"`
	Tuned        RetrievalMetrics `json:"tuned"`
}

// LoadEmbeddingMetrics reads models/<project>/metrics.json from S3 for an
// embedding (contrastive) model.
func LoadEmbeddingMetrics(ctx context.Context, dl S3Downloader, bucket, project string) (*EmbeddingMetrics, error) {
	key := fmt.Sprintf("models/%s/metrics.json", project)
	data, err := dl.DownloadFromS3(ctx, bucket, key)
	if err != nil {
		return nil, fmt.Errorf("downloading metrics.json: %w", err)
	}
	var m EmbeddingMetrics
	if err := json.Unmarshal([]byte(data), &m); err != nil {
		return nil, fmt.Errorf("parsing metrics.json: %w", err)
	}
	return &m, nil
}

// PrintEmbeddingMetrics renders the embedding (contrastive) retrieval report,
// showing the stock encoder baseline next to the domain-tuned result.
func PrintEmbeddingMetrics(m *EmbeddingMetrics) {
	delta := func(tuned, base float64) string {
		d := (tuned - base) * 100
		sign := "+"
		if d < 0 {
			sign = ""
		}
		return fmt.Sprintf("%s%.1f pts", sign, d)
	}
	fmt.Printf("\n  ━━━ Embedding Report (contrastive retriever) ━━━\n")
	fmt.Printf("  Metric        Stock     Tuned     Δ\n")
	fmt.Printf("  Recall@1      %5.1f%%   %5.1f%%   %s\n",
		m.Baseline.Recall1*100, m.Tuned.Recall1*100, delta(m.Tuned.Recall1, m.Baseline.Recall1))
	fmt.Printf("  Recall@5      %5.1f%%   %5.1f%%   %s\n",
		m.Baseline.Recall5*100, m.Tuned.Recall5*100, delta(m.Tuned.Recall5, m.Baseline.Recall5))
	fmt.Printf("  Recall@10     %5.1f%%   %5.1f%%   %s\n",
		m.Baseline.Recall10*100, m.Tuned.Recall10*100, delta(m.Tuned.Recall10, m.Baseline.Recall10))
	fmt.Printf("  MRR           %5.3f    %5.3f    %s\n",
		m.Baseline.MRR, m.Tuned.MRR, delta(m.Tuned.MRR, m.Baseline.MRR))
	fmt.Printf("  Encoder:     %dd embeddings, %d epoch(s), %.0fs CPU train\n",
		m.EmbeddingDim, m.Epochs, m.TrainSeconds)
	fmt.Printf("  Data:        %d pairs / %d eval queries / %d-doc corpus\n",
		m.TrainSamples, m.EvalQueries, m.CorpusSize)
	fmt.Printf("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
}

// RankingMetrics holds recall@k, NDCG@k and MRR for a reranking run.
type RankingMetrics struct {
	Recall1     float64 `json:"recall@1"`
	Recall5     float64 `json:"recall@5"`
	Recall10    float64 `json:"recall@10"`
	NDCG1       float64 `json:"ndcg@1"`
	NDCG5       float64 `json:"ndcg@5"`
	NDCG10      float64 `json:"ndcg@10"`
	MRR         float64 `json:"mrr"`
	EvalQueries int     `json:"eval_queries"`
}

// RerankingMetrics mirrors the metrics.json written by the reranking training
// job (cross-encoder fine-tune): baseline (stock) vs tuned ranking quality.
type RerankingMetrics struct {
	Task         string         `json:"task"`
	TrainSamples int            `json:"train_samples"`
	EvalQueries  int            `json:"eval_queries"`
	CorpusSize   int            `json:"corpus_size"`
	Epochs       int            `json:"epochs"`
	TrainSeconds float64        `json:"train_seconds"`
	Baseline     RankingMetrics `json:"baseline"`
	Tuned        RankingMetrics `json:"tuned"`
}

// LoadRerankingMetrics reads models/<project>/metrics.json from S3 for a
// reranking (cross-encoder) model.
func LoadRerankingMetrics(ctx context.Context, dl S3Downloader, bucket, project string) (*RerankingMetrics, error) {
	key := fmt.Sprintf("models/%s/metrics.json", project)
	data, err := dl.DownloadFromS3(ctx, bucket, key)
	if err != nil {
		return nil, fmt.Errorf("downloading metrics.json: %w", err)
	}
	var m RerankingMetrics
	if err := json.Unmarshal([]byte(data), &m); err != nil {
		return nil, fmt.Errorf("parsing metrics.json: %w", err)
	}
	return &m, nil
}

// PrintRerankingMetrics renders the reranking (cross-encoder) report, showing
// the stock cross-encoder baseline next to the domain-tuned result.
func PrintRerankingMetrics(m *RerankingMetrics) {
	delta := func(tuned, base float64) string {
		d := (tuned - base) * 100
		sign := "+"
		if d < 0 {
			sign = ""
		}
		return fmt.Sprintf("%s%.1f pts", sign, d)
	}
	fmt.Printf("\n  ━━━ Reranking Report (cross-encoder) ━━━\n")
	fmt.Printf("  Metric        Stock     Tuned     Δ\n")
	fmt.Printf("  NDCG@5        %5.3f    %5.3f    %s\n",
		m.Baseline.NDCG5, m.Tuned.NDCG5, delta(m.Tuned.NDCG5, m.Baseline.NDCG5))
	fmt.Printf("  Recall@1      %5.1f%%   %5.1f%%   %s\n",
		m.Baseline.Recall1*100, m.Tuned.Recall1*100, delta(m.Tuned.Recall1, m.Baseline.Recall1))
	fmt.Printf("  Recall@5      %5.1f%%   %5.1f%%   %s\n",
		m.Baseline.Recall5*100, m.Tuned.Recall5*100, delta(m.Tuned.Recall5, m.Baseline.Recall5))
	fmt.Printf("  MRR           %5.3f    %5.3f    %s\n",
		m.Baseline.MRR, m.Tuned.MRR, delta(m.Tuned.MRR, m.Baseline.MRR))
	fmt.Printf("  Model:       cross-encoder, %d epoch(s), %.0fs CPU train\n",
		m.Epochs, m.TrainSeconds)
	fmt.Printf("  Data:        %d pos+neg / %d eval queries / %d-doc corpus\n",
		m.TrainSamples, m.EvalQueries, m.CorpusSize)
	fmt.Printf("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
}
