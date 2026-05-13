// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package data

import (
	"context"
	"fmt"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	sigsyaml "sigs.k8s.io/yaml"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

// Stage creates and runs the data pipeline K8s Job.
func Stage(client *k8s.Client, cfg *config.ExpertConfig, ns string, pc *pipeline.PipelineContext) pipeline.StageFunc {
	return func(ctx context.Context) ([]string, error) {
		cmName := fmt.Sprintf("%s-expert-config", cfg.Project.Name)

		// Serialize expert config to YAML and create ConfigMap
		cfgData, err := marshalExpertConfig(cfg)
		if err != nil {
			return nil, fmt.Errorf("marshaling expert config: %w", err)
		}
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      cmName,
				Namespace: ns,
				Labels: map[string]string{
					"slemify.io/project":           cfg.Project.Name,
					"slemify.io/stage":             "data",
					"app.kubernetes.io/managed-by": "slemify",
				},
			},
			Data: map[string]string{
				"expert.yaml": string(cfgData),
			},
		}
		if err := client.ApplyConfigMap(ctx, cm); err != nil {
			return nil, fmt.Errorf("creating expert config ConfigMap: %w", err)
		}

		// Generate and submit Job
		job := DataJobManifest(cfg, ns, cmName, pc)

		jobName, err := client.SubmitJob(ctx, job)
		if err != nil {
			return nil, fmt.Errorf("submitting data job: %w", err)
		}

		fmt.Printf("  Job submitted: %s\n", jobName)
		fmt.Printf("  Bucket: s3://%s/%s\n", cfg.Data.Bucket, cfg.Data.Path)
		fmt.Printf("  Sources: %d\n", len(cfg.Data.Sources))
		fmt.Printf("  Synthetic: %d pairs via %s\n", cfg.Data.Synthetic.Pairs, cfg.Data.Synthetic.Model)

		if pc.NoWait {
			return []string{fmt.Sprintf("job/%s submitted", jobName)}, nil
		}

		if err := client.WatchJobUntilDone(ctx, jobName); err != nil {
			// Surface container logs on failure
			logs, logErr := client.GetJobPodLogs(ctx, jobName)
			if logErr == nil && logs != "" {
				fmt.Printf("  Container logs:\n%s\n", logs)
			}
			return nil, fmt.Errorf("data job failed: %w", err)
		}

		return []string{
			fmt.Sprintf("s3://%s/%s/processed/train.jsonl", cfg.Data.Bucket, cfg.Project.Name),
			fmt.Sprintf("s3://%s/%s/processed/eval.jsonl", cfg.Data.Bucket, cfg.Project.Name),
		}, nil
	}
}

// DataJobManifest creates the K8s Job manifest for the data pipeline.
func DataJobManifest(cfg *config.ExpertConfig, ns, cmName string, pc *pipeline.PipelineContext) *batchv1.Job {
	backoffLimit := int32(2)
	automountSA := pc.ServiceAccount != "" // Pod Identity needs the SA token

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-data", cfg.Project.Name),
			Namespace: ns,
			Labels: map[string]string{
				"slemify.io/project":           cfg.Project.Name,
				"slemify.io/stage":             "data",
				"app.kubernetes.io/managed-by": "slemify",
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"slemify.io/project": cfg.Project.Name,
						"slemify.io/stage":   "data",
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
							Name:            "data-pipeline",
							Image:           pc.Image("data-pipeline"),
							ImagePullPolicy: corev1.PullAlways,
							Args:            []string{"/config/expert.yaml"},
							SecurityContext: k8s.RestrictedSecurityContext(),
							Env: []corev1.EnvVar{
								{Name: "PYTHONUNBUFFERED", Value: "1"},
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("2Gi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("2Gi"),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "config",
									MountPath: "/config",
									ReadOnly:  true,
								},
								{
									Name:      "tmp",
									MountPath: "/tmp",
								},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "config",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: cmName,
									},
								},
							},
						},
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

// marshalExpertConfig serializes an ExpertConfig to YAML bytes.
func marshalExpertConfig(cfg *config.ExpertConfig) ([]byte, error) {
	return sigsyaml.Marshal(cfg)
}
