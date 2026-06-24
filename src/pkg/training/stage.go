// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package training

import (
	"context"
	"fmt"
	"strconv"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

// Stage creates and runs the Unsloth training K8s Job.
func Stage(client *k8s.Client, cfg *config.ExpertConfig, sized config.SizedConfig, ns string, pc *pipeline.PipelineContext) pipeline.StageFunc {
	return func(ctx context.Context) ([]string, error) {
		// Generate Unsloth training script
		trainScript := UnslothTrainingScript(cfg, sized)

		// Incremental retraining adjustments
		if cfg.Training.Incremental {
			sized.LearningRate = sized.LearningRate * 0.5
			sized.WarmupRatio = 0.05
			if cfg.Training.Epochs == 0 {
				sized.Epochs = 2
			}
			trainScript = UnslothTrainingScript(cfg, sized)
			fmt.Println("  Incremental mode: will resume from checkpoint if available")
		} else {
			// Fresh (non-incremental) run: clear stale checkpoints from prior
			// runs so the trainer's Spot-recovery logic doesn't resume from a
			// checkpoint produced on different data (which would skip training
			// entirely if the old run had >= the new step count). Checkpoints
			// written during THIS run are re-uploaded by the in-pod callback, so
			// within-run Spot interruptions still recover correctly.
			prefix := fmt.Sprintf("models/%s/checkpoint-", cfg.Project.Name)
			if n, err := client.DeleteS3Prefix(ctx, cfg.Data.Bucket, prefix); err != nil {
				fmt.Printf("  ⚠ Could not clear stale checkpoints under %s: %v\n", prefix, err)
			} else if n > 0 {
				fmt.Printf("  Cleared %d stale checkpoint object(s) for a fresh training run\n", n)
			}
		}

		// Create ConfigMap with training script
		cmName := fmt.Sprintf("%s-training-script", cfg.Project.Name)
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      cmName,
				Namespace: ns,
				Labels: map[string]string{
					"slemify.io/project":           cfg.Project.Name,
					"slemify.io/stage":             "training",
					"app.kubernetes.io/managed-by": "slemify",
				},
			},
			Data: map[string]string{
				"train.py": trainScript,
			},
		}
		if err := client.ApplyConfigMap(ctx, cm); err != nil {
			return nil, fmt.Errorf("creating training script ConfigMap: %w", err)
		}

		// Generate and submit the training Job
		job := TrainingJobManifest(cfg, sized, ns, cmName, pc)

		jobName, err := client.SubmitJob(ctx, job)
		if err != nil {
			return nil, fmt.Errorf("submitting training job: %w", err)
		}

		fmt.Printf("  Job submitted: %s\n", jobName)
		fmt.Printf("  Model: %s\n", cfg.Model.Base)
		fmt.Printf("  Engine: Unsloth (QLoRA)\n")
		fmt.Printf("  GPU: %s\n", sized.TrainingGPU)
		fmt.Printf("  Instance: %s\n", sized.TrainingInstance)
		fmt.Printf("  Epochs: %d, LR: %g, Scheduler: %s\n", sized.Epochs, sized.LearningRate, sized.Scheduler)
		fmt.Printf("  Checkpoint every %d steps\n", sized.CheckpointInterval)
		if cfg.Training.Spot {
			fmt.Printf("  Spot enabled (backoffLimit: 3)\n")
		}

		if pc.NoWait {
			return []string{fmt.Sprintf("job/%s submitted", jobName)}, nil
		}

		if err := client.WatchJobUntilDone(ctx, jobName); err != nil {
			logs, logErr := client.GetJobPodLogs(ctx, jobName)
			if logErr == nil && logs != "" {
				fmt.Printf("  Container logs:\n%s\n", logs)
			}
			return nil, fmt.Errorf("training job failed: %w", err)
		}

		// Save training logs to S3 for debugging
		logs, _ := client.GetJobPodLogs(ctx, jobName)
		if logs != "" {
			savedLogsPath := fmt.Sprintf("models/%s/training-logs.txt", cfg.Project.Name)
			if err := client.UploadToS3(ctx, cfg.Data.Bucket, savedLogsPath, []byte(logs)); err != nil {
				fmt.Printf("  ⚠ Could not save training logs: %v\n", err)
			}
		}

		return []string{
			fmt.Sprintf("s3://%s/models/%s/", cfg.Data.Bucket, cfg.Project.Name),
		}, nil
	}
}

// TrainingJobManifest creates the K8s Job manifest for training.
func TrainingJobManifest(cfg *config.ExpertConfig, sized config.SizedConfig, ns, cmName string, pc *pipeline.PipelineContext) *batchv1.Job {
	backoffLimit := int32(6) // Higher for Spot retries + image pull time

	gpuQty := resource.MustParse("1")
	automountSA := pc.ServiceAccount != ""
	noEscalation := false

	// Node selector: always amd64 for GPU training.
	nodeSelector := map[string]string{
		"kubernetes.io/arch":  "amd64",
		"slemify.io/workload": "gpu",
	}

	minGPUMemory := "8192" // default: >8Gi allows T4/A10G/L4/A100
	if sized.TrainingGPUMemoryFloor > 0 {
		minGPUMemory = strconv.Itoa(sized.TrainingGPUMemoryFloor)
	}

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
					Annotations: map[string]string{
						"karpenter.sh/do-not-disrupt": "true",
					},
				},
				Spec: corev1.PodSpec{
					ServiceAccountName:           pc.ServiceAccount,
					AutomountServiceAccountToken: &automountSA,
					RestartPolicy:                corev1.RestartPolicyOnFailure,
					NodeSelector:                 nodeSelector,
					Affinity: &corev1.Affinity{
						NodeAffinity: &corev1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
								NodeSelectorTerms: []corev1.NodeSelectorTerm{
									{
										MatchExpressions: []corev1.NodeSelectorRequirement{
											{
												Key:      "karpenter.k8s.aws/instance-gpu-memory",
												Operator: corev1.NodeSelectorOpGt,
												Values:   []string{minGPUMemory},
											},
										},
									},
								},
							},
						},
					},
					Tolerations: []corev1.Toleration{
						{
							Key:      "nvidia.com/gpu",
							Operator: corev1.TolerationOpExists,
							Effect:   corev1.TaintEffectNoSchedule,
						},
					},
					InitContainers: []corev1.Container{
						{
							Name:  "wait-for-gpu",
							Image: "nvidia/cuda:12.6.3-base-ubuntu24.04",
							Command: []string{"sh", "-c",
								"echo 'Waiting for GPU to be ready...' && " +
									"for i in $(seq 1 60); do " +
									"if nvidia-smi > /dev/null 2>&1; then " +
									"echo \"GPU ready: $(nvidia-smi --query-gpu=name --format=csv,noheader)\"; exit 0; " +
									"fi; echo \"Attempt $i/60: GPU not ready yet\"; sleep 5; done; " +
									"echo 'ERROR: GPU not ready after 5 minutes'; exit 1"},
							SecurityContext: &corev1.SecurityContext{
								AllowPrivilegeEscalation: &noEscalation,
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("128Mi"),
								},
								Limits: corev1.ResourceList{
									"nvidia.com/gpu":      gpuQty,
									corev1.ResourceMemory: resource.MustParse("128Mi"),
								},
							},
						},
						{
							Name:  "download-data",
							Image: "amazon/aws-cli:latest",
							Command: []string{"sh", "-c", fmt.Sprintf(
								"aws s3 cp s3://%s/%s/processed/train.jsonl /data/train.jsonl && aws s3 cp s3://%s/%s/processed/eval.jsonl /data/eval.jsonl && echo 'Downloaded training data'",
								cfg.Data.Bucket, cfg.Project.Name, cfg.Data.Bucket, cfg.Project.Name,
							)},
							SecurityContext: &corev1.SecurityContext{
								AllowPrivilegeEscalation: &noEscalation,
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("256Mi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("256Mi"),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "training-data", MountPath: "/data"},
							},
						},
					},
					Containers: []corev1.Container{
						{
							Name:            "training",
							Image:           "unsloth/unsloth:2026.5.7-pt2.10.0-vllm-0.16.0-cu12.8-studio-release-v0.1.41-beta-2026-MAY-24",
							ImagePullPolicy: corev1.PullAlways,
							Command: []string{"bash", "-c", fmt.Sprintf(`
set -e
LOG_FILE="/tmp/training-pod.log"

# Capture all output to log file AND stdout
exec > >(tee -a "$LOG_FILE") 2>&1

# Upload logs to S3 on exit (success or failure)
upload_logs() {
    echo "Uploading pod logs to S3..."
    pip install -q boto3 2>/dev/null || true
    python3 -c "
import boto3
try:
    boto3.client('s3').upload_file('$LOG_FILE', '%s', 'models/%s/training-pod.log')
    print('Pod logs uploaded to S3')
except Exception as e:
    print(f'Failed to upload logs: {e}')
" || true
}
trap upload_logs EXIT

echo "Starting Unsloth training..."

# Remove broken PPAs that cause Unsloth's apt-get update check to fail with "Err:"
# (Unsloth interprets any "Err:" in apt-get output as "no internet connection")
rm -f /etc/apt/sources.list.d/deadsnakes*.list 2>/dev/null || true

# Run the training script (mounted from ConfigMap)
python3 /scripts/train.py

# Upload trained adapter and GGUF directly to S3 via boto3
echo "Installing boto3 for S3 upload..."
pip install boto3 || { echo "ERROR: Failed to install boto3"; exit 1; }
python3 -c "
import os, boto3
s3 = boto3.client('s3')
bucket = '%s'
prefix = 'models/%s'
output_dir = '/tmp/unsloth-output'
gguf_name = '%s'

print(f'Uploading results to s3://{bucket}/{prefix}/')
for fname in os.listdir(output_dir):
    fpath = os.path.join(output_dir, fname)
    if os.path.isfile(fpath):
        key = f'{prefix}/{fname}'
        print(f'  {fname} ({os.path.getsize(fpath) / 1048576:.1f} MB)')
        s3.upload_file(fpath, bucket, key)

# Unsloth saves GGUF to a separate _gguf directory
for d in [output_dir + '_gguf', output_dir]:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith('.gguf'):
                src = os.path.join(d, f)
                key = f'{prefix}/{gguf_name}'
                print(f'  GGUF: {f} ({os.path.getsize(src) / 1048576:.1f} MB) -> {gguf_name}')
                s3.upload_file(src, bucket, key)
                break
        else:
            continue
        break

print('Upload complete')
"

echo "Training complete"
`, cfg.Data.Bucket, cfg.Project.Name, cfg.Data.Bucket, cfg.Project.Name, cfg.Model.GGUFFilename())},
							SecurityContext: &corev1.SecurityContext{
								AllowPrivilegeEscalation: &noEscalation,
								RunAsUser:                func() *int64 { u := int64(0); return &u }(),
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("48Gi"),
								},
								Limits: corev1.ResourceList{
									"nvidia.com/gpu":      gpuQty,
									corev1.ResourceMemory: resource.MustParse("48Gi"),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "scripts", MountPath: "/scripts", ReadOnly: true},
								{Name: "training-data", MountPath: "/data"},
								{Name: "workspace", MountPath: "/workspace/output"},
								{Name: "tmp", MountPath: "/tmp"},
							},
							Env: []corev1.EnvVar{
								{
									Name:  "HF_HOME",
									Value: "/tmp/hf-cache",
								},
								{
									Name:  "S3_CHECKPOINT_PATH",
									Value: fmt.Sprintf("s3://%s/checkpoints/%s/", cfg.Data.Bucket, cfg.Project.Name),
								},
								{
									Name:  "S3_BUCKET",
									Value: cfg.Data.Bucket,
								},
								{
									Name:  "S3_CHECKPOINT_PREFIX",
									Value: fmt.Sprintf("models/%s/", cfg.Project.Name),
								},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "scripts",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: cmName,
									},
								},
							},
						},
						{
							Name: "training-data",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{},
							},
						},
						{
							Name:         "workspace",
							VolumeSource: WorkspaceVolumeSource(),
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

// WorkspaceVolumeSource returns an S3-backed PVC volume if configured,
// otherwise falls back to emptyDir.
func WorkspaceVolumeSource() corev1.VolumeSource {
	return corev1.VolumeSource{
		EmptyDir: &corev1.EmptyDirVolumeSource{},
	}
}

// FindLatestCheckpoint returns the S3 path to the latest checkpoint directory, or empty string if none found.
func FindLatestCheckpoint(ctx context.Context, client *k8s.Client, cfg *config.ExpertConfig) string {
	adapterPath := fmt.Sprintf("models/%s/adapter_config.json", cfg.Project.Name)
	_, err := client.DownloadFromS3(ctx, cfg.Data.Bucket, adapterPath)
	if err != nil {
		return "" // no previous training
	}

	for step := 10000; step >= 100; step -= 100 {
		checkpointPath := fmt.Sprintf("models/%s/checkpoint-%d/trainer_state.json", cfg.Project.Name, step)
		_, err := client.DownloadFromS3(ctx, cfg.Data.Bucket, checkpointPath)
		if err == nil {
			return fmt.Sprintf("/workspace/output/checkpoint-%d", step)
		}
	}

	return "" // adapter exists but no numbered checkpoints
}
