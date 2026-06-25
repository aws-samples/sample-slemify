// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package training

import (
	"context"
	"fmt"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

// ConvertStage runs the CPU-only GGUF conversion Job for generative experts.
// Slemify serves generative models stock (no fine-tuning): the Job downloads the
// HuggingFace base model, converts it to a GGUF (f16), quantizes it with
// llama.cpp, and uploads model-<quant>.gguf to S3 where the serving pod reads it.
// Knowledge comes from RAG at serving time, not from training. There is no GPU
// in this path.
func ConvertStage(client *k8s.Client, cfg *config.ExpertConfig, sized config.SizedConfig, ns string, pc *pipeline.PipelineContext) pipeline.StageFunc {
	return func(ctx context.Context) ([]string, error) {
		job := ConvertJobManifest(cfg, sized, ns, pc)

		jobName, err := client.SubmitJob(ctx, job)
		if err != nil {
			return nil, fmt.Errorf("submitting convert job: %w", err)
		}

		fmt.Printf("  Job submitted: %s\n", jobName)
		fmt.Printf("  Base model: %s\n", cfg.Model.Base)
		fmt.Printf("  Engine: download base -> convert to GGUF -> quantize (CPU, no GPU)\n")
		fmt.Printf("  Quantization: %s\n", cfg.Model.QuantizeLabel())
		fmt.Printf("  Instance: %s\n", sized.TrainingInstance)
		fmt.Printf("  Memory: %s, Ephemeral storage: %s\n", sized.ConvertMemory, sized.ConvertEphemeralStorage)

		if pc.NoWait {
			return []string{fmt.Sprintf("job/%s submitted", jobName)}, nil
		}

		if err := client.WatchJobUntilDone(ctx, jobName); err != nil {
			logs, logErr := client.GetJobPodLogs(ctx, jobName)
			if logErr == nil && logs != "" {
				fmt.Printf("  Container logs:\n%s\n", logs)
			}
			return nil, fmt.Errorf("convert job failed: %w", err)
		}

		// Save conversion logs to S3 for debugging.
		logs, _ := client.GetJobPodLogs(ctx, jobName)
		if logs != "" {
			savedLogsPath := fmt.Sprintf("models/%s/convert-logs.txt", cfg.Project.Name)
			if err := client.UploadToS3(ctx, cfg.Data.Bucket, savedLogsPath, []byte(logs)); err != nil {
				fmt.Printf("  ⚠ Could not save convert logs: %v\n", err)
			}
		}

		return []string{
			fmt.Sprintf("s3://%s/models/%s/%s", cfg.Data.Bucket, cfg.Project.Name, cfg.Model.GGUFFilename()),
		}, nil
	}
}

// ConvertJobManifest builds the CPU K8s Job for the GGUF conversion pipeline.
// It runs the gguf-convert image on the SLM (CPU) node pool — no GPU nodeSelector,
// no nvidia toleration, no GPU resources. The Job needs S3 write (model upload)
// and HuggingFace pull access, both provided by the Pod Identity service account.
func ConvertJobManifest(cfg *config.ExpertConfig, sized config.SizedConfig, ns string, pc *pipeline.PipelineContext) *batchv1.Job {
	backoffLimit := int32(3)
	automountSA := pc.ServiceAccount != ""

	// Memory and ephemeral storage are sized for the model: an 8B f16 GGUF is
	// ~16GB, plus the downloaded weights and the quantized output all live on
	// the node's ephemeral disk during the run.
	convertMem := sized.ConvertMemory
	if convertMem == "" {
		convertMem = "48Gi"
	}
	ephemeral := sized.ConvertEphemeralStorage
	if ephemeral == "" {
		ephemeral = "120Gi"
	}

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-training", cfg.Project.Name),
			Namespace: ns,
			Labels: map[string]string{
				"slemify.io/project":           cfg.Project.Name,
				"slemify.io/stage":             "convert",
				"app.kubernetes.io/managed-by": "slemify",
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"slemify.io/project": cfg.Project.Name,
						"slemify.io/stage":   "convert",
					},
					Annotations: map[string]string{
						"karpenter.sh/do-not-disrupt": "true",
					},
				},
				Spec: corev1.PodSpec{
					ServiceAccountName:           pc.ServiceAccount,
					AutomountServiceAccountToken: &automountSA,
					SecurityContext:              k8s.RestrictedPodSecurityContext(),
					RestartPolicy:                corev1.RestartPolicyOnFailure,
					// CPU pool only — the conversion runs entirely on CPU. Pin to
					// on-demand: the convert Job is a one-shot, bandwidth-heavy run
					// (downloads ~16GB of weights). A Spot reclaim mid-run forces a
					// full re-download, so the small on-demand premium is worth the
					// determinism here.
					NodeSelector: map[string]string{
						"slemify.io/workload":        "slm",
						"karpenter.sh/capacity-type": "on-demand",
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
							Name:            "convert",
							Image:           pc.Image("gguf-convert"),
							ImagePullPolicy: corev1.PullAlways,
							SecurityContext: k8s.RestrictedSecurityContext(),
							Env: []corev1.EnvVar{
								{Name: "PYTHONUNBUFFERED", Value: "1"},
								{Name: "BASE_MODEL", Value: cfg.Model.Base},
								{Name: "QUANTIZE", Value: cfg.Model.QuantizeType()},
								{Name: "S3_BUCKET", Value: cfg.Data.Bucket},
								{Name: "PROJECT", Value: cfg.Project.Name},
								{Name: "GGUF_FILENAME", Value: cfg.Model.GGUFFilename()},
								{Name: "HF_HOME", Value: "/tmp/hf-cache"},
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceMemory:           resource.MustParse(convertMem),
									corev1.ResourceCPU:              resource.MustParse("4"),
									corev1.ResourceEphemeralStorage: resource.MustParse(ephemeral),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory:           resource.MustParse(convertMem),
									corev1.ResourceEphemeralStorage: resource.MustParse(ephemeral),
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
