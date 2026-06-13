// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"context"
	"fmt"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
	"github.com/aws-samples/sample-slemify/pkg/report"
)

// Stage creates and applies the inference serving manifests:
// Karpenter NodePool, Deployment, Service, and PDB.
// After deployment is ready, submits a report Job that evaluates the model.
func Stage(client *k8s.Client, cfg *config.ExpertConfig, sized config.SizedConfig, ns string, pc *pipeline.PipelineContext) pipeline.StageFunc {
	return func(ctx context.Context) ([]string, error) {
		var manifests *InferenceManifests
		if cfg.Project.IsEncoderHead() || cfg.Project.IsEmbedding() {
			manifests = GenerateClassifierInferenceManifests(cfg, sized, ns, pc)
		} else {
			manifests = GenerateInferenceManifests(cfg, sized, ns, pc)
		}

		fmt.Printf("  Instance: %s\n", sized.InferenceInstance)

		// Apply S3 mount PV/PVC if Mountpoint CSI driver is being used.
		// Only the generative (GGUF) path mounts a model from S3; the encoder
		// family loads its ONNX artifacts directly via the AWS SDK at startup.
		if pc.UseS3Mount && !cfg.Project.IsEncoderHead() && !cfg.Project.IsEmbedding() {
			fmt.Printf("  Setting up S3 mount for model (Mountpoint CSI driver)...\n")
			s3Manifests := S3MountManifests(cfg.Project.Name, cfg.Data.Bucket, ns)
			for _, doc := range pipeline.SplitYAMLDocs(s3Manifests) {
				if err := client.ApplyYAML(ctx, []byte(doc)); err != nil {
					return nil, fmt.Errorf("applying S3 mount manifest: %w", err)
				}
			}
		}

		// Force pod restart on every deploy so the model is re-read
		// (S3 mount: Mountpoint re-reads from S3; download mode: init container re-downloads).
		if manifests.Deployment.Spec.Template.Annotations == nil {
			manifests.Deployment.Spec.Template.Annotations = map[string]string{}
		}
		manifests.Deployment.Spec.Template.Annotations["slemify.io/deployed-at"] = time.Now().UTC().Format(time.RFC3339)

		if err := client.ApplyDeployment(ctx, manifests.Deployment); err != nil {
			return nil, fmt.Errorf("applying inference Deployment: %w", err)
		}

		if err := client.ApplyService(ctx, manifests.Service); err != nil {
			return nil, fmt.Errorf("applying inference Service: %w", err)
		}

		if err := client.ApplyPDB(ctx, manifests.PodDisruptionBudget); err != nil {
			return nil, fmt.Errorf("applying inference PDB: %w", err)
		}

		fmt.Printf("  Waiting for Deployment readiness...\n")
		depName := fmt.Sprintf("%s-inference", cfg.Project.Name)
		if err := client.WaitForDeploymentReady(ctx, depName, 5*time.Minute); err != nil {
			return nil, fmt.Errorf("inference Deployment not ready: %w", err)
		}

		endpoint := fmt.Sprintf("http://%s-inference.%s.svc.cluster.local:8080", cfg.Project.Name, ns)

		if cfg.Project.IsEncoderHead() || cfg.Project.IsEmbedding() {
			// Encoder family metrics were computed by the training job and written
			// to metrics.json. Surface those instead of the generative
			// LLM-as-judge report. Scoring uses regression metrics (MAE/R²);
			// embedding uses retrieval metrics (recall@k/MRR, baseline vs tuned);
			// classification uses exact-match accuracy + per-class P/R/F1.
			if cfg.Project.IsEmbedding() {
				if m, err := report.LoadEmbeddingMetrics(ctx, client, cfg.Data.Bucket, cfg.Project.Name); err != nil {
					fmt.Printf("  ⚠ Could not load embedding metrics: %v\n", err)
				} else {
					report.PrintEmbeddingMetrics(m)
				}
			} else if cfg.Project.IsReranking() {
				if m, err := report.LoadRerankingMetrics(ctx, client, cfg.Data.Bucket, cfg.Project.Name); err != nil {
					fmt.Printf("  ⚠ Could not load reranking metrics: %v\n", err)
				} else {
					report.PrintRerankingMetrics(m)
				}
			} else if cfg.Project.IsScoring() {
				if m, err := report.LoadScoringMetrics(ctx, client, cfg.Data.Bucket, cfg.Project.Name); err != nil {
					fmt.Printf("  ⚠ Could not load scoring metrics: %v\n", err)
				} else {
					report.PrintScoringMetrics(m)
				}
			} else if m, err := report.LoadClassificationMetrics(ctx, client, cfg.Data.Bucket, cfg.Project.Name); err != nil {
				fmt.Printf("  ⚠ Could not load classification metrics: %v\n", err)
			} else {
				report.PrintEncoderHeadMetrics(m)
			}
			return []string{endpoint}, nil
		}

		// Run classification report as a K8s Job (direct access to inference service)
		fmt.Printf("  Running classification report (in-cluster)...\n")
		classReport, err := runReportJob(ctx, client, cfg, ns, 100, pc)
		if err != nil {
			fmt.Printf("  ⚠ Report failed: %v\n", err)
		} else if classReport != nil {
			report.PrintReport(classReport)
		}

		return []string{endpoint}, nil
	}
}

// runReportJob submits a K8s Job that evaluates the model and uploads
// the HTML report to S3.
func runReportJob(
	ctx context.Context,
	client *k8s.Client,
	cfg *config.ExpertConfig,
	ns string,
	maxSamples int,
	pc *pipeline.PipelineContext,
) (*report.ClassificationReport, error) {
	inferenceEndpoint := fmt.Sprintf("http://%s-inference.%s.svc.cluster.local:8080", cfg.Project.Name, ns)

	// Submit the report Job (report.py is baked into the container image)
	job := ReportJobManifest(cfg, ns, inferenceEndpoint, maxSamples, pc)
	jobName, err := client.SubmitJob(ctx, job)
	if err != nil {
		return nil, fmt.Errorf("submitting report job: %w", err)
	}
	fmt.Printf("  Report job submitted: %s\n", jobName)

	if pc.NoWait {
		return nil, nil
	}

	// Wait for completion
	if err := client.WatchJobUntilDone(ctx, jobName); err != nil {
		logs, logErr := client.GetJobPodLogs(ctx, jobName)
		if logErr == nil && logs != "" {
			fmt.Printf("  Report logs:\n%s\n", logs)
		}
		return nil, fmt.Errorf("report job failed: %w", err)
	}

	fmt.Printf("  Report available at: s3://%s/%s/report/report.html\n", cfg.Data.Bucket, cfg.Project.Name)
	return nil, nil
}

// ReportJobManifest creates the K8s Job manifest for the classification report.
// The report.py script is baked into the data-pipeline container image.
// Configuration is passed via environment variables.
func ReportJobManifest(cfg *config.ExpertConfig, ns, inferenceEndpoint string, maxSamples int, pc *pipeline.PipelineContext) *batchv1.Job {
	backoffLimit := int32(1)
	automountSA := pc.ServiceAccount != ""

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-report", cfg.Project.Name),
			Namespace: ns,
			Labels: map[string]string{
				"slemify.io/project":           cfg.Project.Name,
				"slemify.io/stage":             "report",
				"app.kubernetes.io/managed-by": "slemify",
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"slemify.io/project": cfg.Project.Name,
						"slemify.io/stage":   "report",
					},
				},
				Spec: corev1.PodSpec{
					ServiceAccountName:           pc.ServiceAccount,
					AutomountServiceAccountToken: &automountSA,
					SecurityContext:              k8s.RestrictedPodSecurityContext(),
					RestartPolicy:                corev1.RestartPolicyNever,
					NodeSelector: map[string]string{
						"slemify.io/workload": "slm",
					},
					Tolerations: []corev1.Toleration{
						{
							Key:      "slemify.io/slm",
							Operator: corev1.TolerationOpExists,
							Effect:   corev1.TaintEffectNoSchedule,
						},
					},
					Containers: []corev1.Container{
						{
							Name:            "report",
							Image:           pc.Image("data-pipeline"),
							ImagePullPolicy: corev1.PullAlways,
							Command:         []string{"python3", "report.py"},
							SecurityContext: k8s.RestrictedSecurityContext(),
							Env: []corev1.EnvVar{
								{Name: "PYTHONUNBUFFERED", Value: "1"},
								{Name: "BUCKET", Value: cfg.Data.Bucket},
								{Name: "PROJECT", Value: cfg.Project.Name},
								{Name: "INFERENCE_ENDPOINT", Value: inferenceEndpoint},
								{Name: "BEDROCK_MODEL", Value: cfg.Data.Synthetic.Model},
								{Name: "MAX_SAMPLES", Value: fmt.Sprintf("%d", maxSamples)},
								{Name: "TOOL_NAME", Value: cfg.Project.Name},
								{Name: "TOOL_DESC", Value: cfg.Project.Domain},
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("512Mi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("512Mi"),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "tmp", MountPath: "/tmp"},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "tmp",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			},
		},
	}
}

