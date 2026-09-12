// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"context"
	"fmt"
	"strings"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
	"github.com/aws-samples/sample-slemify/pkg/pricing"
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

		// One report Job for every task family. It measures against the served
		// endpoint, adds the baseline and the split the training job recorded,
		// and writes report.json plus report.html to S3 for `slemify report`.
		// If the Job fails, fall back to the metrics the training job wrote so
		// the encoder-family numbers are never hidden by a report problem.
		fmt.Printf("  Running the report (in-cluster)...\n")
		if err := runReportJob(ctx, client, cfg, ns, pc); err != nil {
			fmt.Printf("  ⚠ Report failed: %v\n", err)
			if cfg.Project.IsEncoderHead() || cfg.Project.IsEmbedding() {
				printTrainingMetrics(ctx, client, cfg)
			}
		} else if !pc.NoWait {
			key := fmt.Sprintf("%s/report/report.json", cfg.Project.Name)
			if data, err := client.DownloadFromS3(ctx, cfg.Data.Bucket, key); err != nil {
				fmt.Printf("  ⚠ Could not read %s: %v\n", key, err)
			} else if s, err := report.ParseSummary(data); err != nil {
				fmt.Printf("  ⚠ %v\n", err)
			} else {
				report.PrintSummary(s)
				fmt.Printf("  Full report: slemify report --config <expert.yaml>  (s3://%s/%s/report/report.html)\n",
					cfg.Data.Bucket, cfg.Project.Name)
			}
		}

		return []string{endpoint}, nil
	}
}

// printTrainingMetrics prints the metrics.json the training job wrote.
func printTrainingMetrics(ctx context.Context, client *k8s.Client, cfg *config.ExpertConfig) {
	switch {
	case cfg.Project.IsEmbedding():
		if m, err := report.LoadEmbeddingMetrics(ctx, client, cfg.Data.Bucket, cfg.Project.Name); err != nil {
			fmt.Printf("  ⚠ Could not load embedding metrics: %v\n", err)
		} else {
			report.PrintEmbeddingMetrics(m)
		}
	case cfg.Project.IsScoring():
		if m, err := report.LoadScoringMetrics(ctx, client, cfg.Data.Bucket, cfg.Project.Name); err != nil {
			fmt.Printf("  ⚠ Could not load scoring metrics: %v\n", err)
		} else {
			report.PrintScoringMetrics(m)
		}
	case cfg.Project.IsExtraction():
		if m, err := report.LoadExtractionMetrics(ctx, client, cfg.Data.Bucket, cfg.Project.Name); err != nil {
			fmt.Printf("  ⚠ Could not load extraction metrics: %v\n", err)
		} else {
			report.PrintExtractionMetrics(m)
		}
	default:
		if m, err := report.LoadClassificationMetrics(ctx, client, cfg.Data.Bucket, cfg.Project.Name); err != nil {
			fmt.Printf("  ⚠ Could not load classification metrics: %v\n", err)
		} else {
			report.PrintEncoderHeadMetrics(m)
		}
	}
}

// runReportJob submits the report Job and waits for it. The Job needs to know
// which node the inference pod landed on (for the bandwidth estimate and the
// hourly rate), which only the CLI can look up, so that is resolved here and
// passed in as environment.
func runReportJob(ctx context.Context, client *k8s.Client, cfg *config.ExpertConfig, ns string, pc *pipeline.PipelineContext) error {
	inferenceEndpoint := fmt.Sprintf("http://%s-inference.%s.svc.cluster.local:8080", cfg.Project.Name, ns)
	env := ReportEnv{}
	if info, err := client.InferenceNodeInfo(ctx, cfg.Project.Name); err != nil {
		fmt.Printf("  (node details unavailable: %v)\n", err)
	} else {
		env.InstanceType, env.VCPUs = info.InstanceType, info.VCPUs
		if price, err := pricing.OnDemandHourlyUSD(ctx, pc.Region, info.InstanceType); err != nil {
			fmt.Printf("  (on-demand price unavailable: %v)\n", err)
		} else {
			env.HourlyUSD = price
		}
	}
	job := ReportJobManifest(cfg, ns, inferenceEndpoint, pc, env)
	jobName, err := client.SubmitJob(ctx, job)
	if err != nil {
		return fmt.Errorf("submitting report job: %w", err)
	}
	fmt.Printf("  Report job submitted: %s\n", jobName)
	if pc.NoWait {
		return nil
	}
	if err := client.WatchJobUntilDone(ctx, jobName); err != nil {
		logs, logErr := client.GetJobPodLogs(ctx, jobName)
		if logErr == nil && logs != "" {
			fmt.Printf("  Report logs:\n%s\n", logs)
		}
		return fmt.Errorf("report job failed: %w", err)
	}
	return nil
}

// ReportEnv is what the CLI resolves for the report Job about the node the
// inference pod runs on. Zero values mean "unknown"; the report says so.
type ReportEnv struct {
	InstanceType string
	VCPUs        int
	HourlyUSD    float64
}

// ReportJobManifest creates the K8s Job that runs containers/data-pipeline/report.py
// against the served model. Configuration is passed as environment variables;
// the optional items that cost frontier-model calls (the zero-shot baseline for
// a classifier, the grounded evaluation for a generator) come from
// cfg.Report and are off unless set.
func ReportJobManifest(cfg *config.ExpertConfig, ns, inferenceEndpoint string, pc *pipeline.PipelineContext, env ReportEnv) *batchv1.Job {
	labels := strings.Join(cfg.Project.FlatLabels(), ",")
	casesKey := ""
	if cfg.Report.Cases != "" {
		casesKey = strings.TrimSuffix(cfg.Data.Path, "/") + "/" + strings.TrimPrefix(cfg.Report.Cases, "/")
	}
	repeat := cfg.Report.Repeat
	if repeat == 0 {
		repeat = 2
	}
	bedrockModel := cfg.Report.Model
	if bedrockModel == "" {
		bedrockModel = cfg.Data.Synthetic.Model
	}
	if bedrockModel == "" && cfg.Data.Evaluation != nil {
		bedrockModel = cfg.Data.Evaluation.Model
	}
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
								{Name: "TASK", Value: cfg.Project.Task},
								{Name: "BEDROCK_MODEL", Value: bedrockModel},
								{Name: "MAX_SAMPLES", Value: "100"},
								{Name: "TOOL_DESC", Value: cfg.Project.Domain},
								{Name: "LABELS", Value: labels},
								{Name: "LLM_BASELINE", Value: fmt.Sprintf("%t", cfg.Report.LLMBaseline)},
								{Name: "CASES_KEY", Value: casesKey},
								{Name: "REPEAT", Value: fmt.Sprintf("%d", repeat)},
								{Name: "INSTANCE_TYPE", Value: env.InstanceType},
								{Name: "INSTANCE_VCPUS", Value: fmt.Sprintf("%d", env.VCPUs)},
								{Name: "INSTANCE_HOURLY_USD", Value: fmt.Sprintf("%.4f", env.HourlyUSD)},
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
