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

// ClassifierStage runs the encoder-head classifier training as a CPU K8s Job.
// It embeds training inputs against the managed encoder service and fits a
// classifier head — no GPU, no GGUF. Artifacts (head.json, labels.json,
// metrics.json) are uploaded to S3 by the job itself.
func ClassifierStage(client *k8s.Client, cfg *config.ExpertConfig, ns string, pc *pipeline.PipelineContext) pipeline.StageFunc {
	return func(ctx context.Context) ([]string, error) {
		job := ClassifierJobManifest(cfg, ns, pc)

		jobName, err := client.SubmitJob(ctx, job)
		if err != nil {
			return nil, fmt.Errorf("submitting classifier training job: %w", err)
		}

		fmt.Printf("  Job submitted: %s\n", jobName)
		fmt.Printf("  Encoder: %s\n", cfg.Model.Base)
		fmt.Printf("  Head: %s\n", cfg.Model.HeadType())
		fmt.Printf("  Engine: encoder-head (CPU, no GPU)\n")

		if pc.NoWait {
			return []string{fmt.Sprintf("job/%s submitted", jobName)}, nil
		}

		if err := client.WatchJobUntilDone(ctx, jobName); err != nil {
			logs, logErr := client.GetJobPodLogs(ctx, jobName)
			if logErr == nil && logs != "" {
				fmt.Printf("  Container logs:\n%s\n", logs)
			}
			return nil, fmt.Errorf("classifier training job failed: %w", err)
		}

		// Surface the training logs (they include the accuracy line).
		logs, _ := client.GetJobPodLogs(ctx, jobName)
		if logs != "" {
			savedLogsPath := fmt.Sprintf("models/%s/training-logs.txt", cfg.Project.Name)
			_ = client.UploadToS3(ctx, cfg.Data.Bucket, savedLogsPath, []byte(logs))
		}

		return []string{
			fmt.Sprintf("s3://%s/models/%s/head.json", cfg.Data.Bucket, cfg.Project.Name),
		}, nil
	}
}

// ClassifierJobManifest builds the CPU K8s Job for encoder-head training.
// It reuses the data-pipeline image (which bundles classifier_train.py and
// scikit-learn) and runs on the SLM (CPU) node pool.
func ClassifierJobManifest(cfg *config.ExpertConfig, ns string, pc *pipeline.PipelineContext) *batchv1.Job {
	backoffLimit := int32(2)
	automountSA := pc.ServiceAccount != ""

	encoderURL := fmt.Sprintf("http://%s-encoder.%s.svc.cluster.local:8080", cfg.Project.Name, ns)

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-training", cfg.Project.Name),
			Namespace: ns,
			Labels: map[string]string{
				"slemify.io/project":           cfg.Project.Name,
				"slemify.io/stage":             "training",
				"app.kubernetes.io/managed-by": "slemify",
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"slemify.io/project": cfg.Project.Name,
						"slemify.io/stage":   "training",
					},
				},
				Spec: corev1.PodSpec{
					ServiceAccountName:           pc.ServiceAccount,
					AutomountServiceAccountToken: &automountSA,
					SecurityContext:              k8s.RestrictedPodSecurityContext(),
					RestartPolicy:                corev1.RestartPolicyOnFailure,
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
							Name:            "classifier-training",
							Image:           pc.Image("data-pipeline"),
							ImagePullPolicy: corev1.PullAlways,
							Command:         []string{"python3", "classifier_train.py"},
							SecurityContext: k8s.RestrictedSecurityContext(),
							Env: []corev1.EnvVar{
								{Name: "PYTHONUNBUFFERED", Value: "1"},
								{Name: "S3_BUCKET", Value: cfg.Data.Bucket},
								{Name: "PROJECT", Value: cfg.Project.Name},
								{Name: "ENCODER_URL", Value: encoderURL},
								{Name: "HEAD", Value: cfg.Model.HeadType()},
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("2Gi"),
									corev1.ResourceCPU:    resource.MustParse("1"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("2Gi"),
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
