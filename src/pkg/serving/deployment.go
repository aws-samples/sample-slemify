// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Package serving handles inference deployment generation including
// llama.cpp Deployment, Karpenter NodePool, and PDB.
package serving

import (
	"fmt"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

// InferenceManifests holds all K8s manifests for the inference serving layer.
type InferenceManifests struct {
	Deployment          *appsv1.Deployment
	Service             *corev1.Service
	PodDisruptionBudget *policyv1.PodDisruptionBudget
}

// GenerateInferenceManifests creates the Deployment, Service, and PDB for llama.cpp inference.
func GenerateInferenceManifests(cfg *config.ExpertConfig, sized config.SizedConfig, ns string, pc *pipeline.PipelineContext) *InferenceManifests {
	labels := map[string]string{
		"slemify.io/project":           cfg.Project.Name,
		"slemify.io/stage":             "serving",
		"app.kubernetes.io/managed-by": "slemify",
	}
	selectorLabels := map[string]string{
		"slemify.io/project": cfg.Project.Name,
		"slemify.io/stage":   "serving",
	}

	return &InferenceManifests{
		Deployment:          inferenceDeployment(cfg, sized, ns, labels, selectorLabels, pc),
		Service:             inferenceService(cfg, ns, labels, selectorLabels),
		PodDisruptionBudget: inferencePDB(cfg, ns, labels, selectorLabels),
	}
}

func inferenceDeployment(cfg *config.ExpertConfig, sized config.SizedConfig, ns string, labels, selectorLabels map[string]string, pc *pipeline.PipelineContext) *appsv1.Deployment {
	replicas := int32(1)
	automountSA := pc.ServiceAccount != ""

	// Model path depends on whether we use S3 mount or init container download.
	// With S3 mount: the PV prefix is models/<project>/, so the GGUF file appears
	// at /models/<gguf-filename>. llama.cpp reads it via mmap directly from S3.
	// With init container: the file is downloaded to /models/model.gguf.
	modelPath := "/models/model.gguf"
	if pc.UseS3Mount {
		modelPath = fmt.Sprintf("/models/%s", cfg.Model.GGUFFilename())
	}

	var initContainers []corev1.Container
	var modelVolume corev1.Volume

	if pc.UseS3Mount {
		// S3 mount: PVC backed by Mountpoint for Amazon S3 CSI driver.
		// No init container needed — llama.cpp mmaps the file directly from S3.
		modelVolume = corev1.Volume{
			Name: "model-storage",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: S3ModelVolumeName(cfg.Project.Name),
					ReadOnly:  true,
				},
			},
		}
	} else {
		// Download mode: init container copies GGUF from S3 to emptyDir.
		initContainers = []corev1.Container{
			{
				Name:  "model-loader",
				Image: "amazon/aws-cli:2.22.35",
				// Use explicit args instead of sh -c to prevent injection
				Command: []string{"aws", "s3", "cp",
					fmt.Sprintf("s3://%s/models/%s/%s", cfg.Data.Bucket, cfg.Project.Name, cfg.Model.GGUFFilename()),
					"/models/model.gguf"},
				SecurityContext: &corev1.SecurityContext{
					AllowPrivilegeEscalation: func() *bool { b := false; return &b }(),
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
					{Name: "model-storage", MountPath: "/models"},
				},
			},
		}
		modelVolume = corev1.Volume{
			Name: "model-storage",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		}
	}

	// Build llama.cpp args with the correct model path
	llamaArgs := llamaCppArgs(cfg, modelPath, sized)

	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-inference", cfg.Project.Name),
			Namespace: ns,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: selectorLabels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: selectorLabels,
				},
				Spec: corev1.PodSpec{
					AutomountServiceAccountToken: &automountSA,
					ServiceAccountName:           pc.ServiceAccount,
					NodeSelector:                 map[string]string{"slemify.io/workload": "slm"},
					Tolerations: []corev1.Toleration{
						{
							Key:      "slemify.io/slm",
							Operator: corev1.TolerationOpExists,
							Effect:   corev1.TaintEffectNoSchedule,
						},
					},
					InitContainers: initContainers,
					Containers: []corev1.Container{
						{
							Name:  "llama-cpp",
							Image: "ghcr.io/ggml-org/llama.cpp:server",
							Args:  llamaArgs,
							SecurityContext: &corev1.SecurityContext{
								AllowPrivilegeEscalation: func() *bool { b := false; return &b }(),
							},
							Ports: []corev1.ContainerPort{
								{Name: "http", ContainerPort: 8080, Protocol: corev1.ProtocolTCP},
							},
							Lifecycle: &corev1.Lifecycle{
								PostStart: &corev1.LifecycleHandler{
									Exec: &corev1.ExecAction{
										Command: warmupCommand(),
									},
								},
							},
							ReadinessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/health",
										Port: intstr.FromInt32(8080),
									},
								},
								InitialDelaySeconds: 10,
								PeriodSeconds:       5,
							},
							LivenessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/health",
										Port: intstr.FromInt32(8080),
									},
								},
								InitialDelaySeconds: 120,
								PeriodSeconds:       15,
								FailureThreshold:    5,
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse(sized.InferenceCPU),
									corev1.ResourceMemory: resource.MustParse(sized.InferenceMemory),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse(sized.InferenceMemory),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "model-storage", MountPath: "/models", ReadOnly: true},
								{Name: "tmp", MountPath: "/tmp"},
							},
						},
					},
					Volumes: []corev1.Volume{
						modelVolume,
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

func inferenceService(cfg *config.ExpertConfig, ns string, labels, selectorLabels map[string]string) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-inference", cfg.Project.Name),
			Namespace: ns,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Type:     corev1.ServiceTypeClusterIP,
			Selector: selectorLabels,
			Ports: []corev1.ServicePort{
				{
					Name:       "http",
					Port:       8080,
					TargetPort: intstr.FromInt32(8080),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}
}

func inferencePDB(cfg *config.ExpertConfig, ns string, labels, selectorLabels map[string]string) *policyv1.PodDisruptionBudget {
	minAvailable := intstr.FromInt32(1)
	return &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-inference", cfg.Project.Name),
			Namespace: ns,
			Labels:    labels,
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			MinAvailable: &minAvailable,
			Selector: &metav1.LabelSelector{
				MatchLabels: selectorLabels,
			},
		},
	}
}

// warmupCommand returns the postStart lifecycle hook command that sends a dummy
// completion request to the llama.cpp server after it starts. This forces the
// model weights into memory and primes the KV cache, eliminating the cold-start
// penalty on the first real user request.
// The script polls /health until the server is ready, then sends a completion
// request with a representative system prompt to pre-compute the KV cache prefix
// that real queries will share.
func warmupCommand() []string {
	return []string{
		"/bin/sh", "-c",
		// Write the warmup payload to a file to avoid shell quoting issues,
		// then use wget --post-file to send it.
		`printf '{"messages":[{"role":"user","content":"You are a Kubernetes autoscaling auditor. Answer ONLY based on the reference documentation below. Do NOT invent fields, behaviors, or modes not in the docs. If the docs do not cover something, say so. State what is correct, why, and provide a fix if needed.\n\n--- REFERENCE DOCUMENTATION ---\nNodePool limits define the maximum amount of resources Karpenter can provision. When limits are reached, Karpenter will not launch new nodes.\n--- END REFERENCE ---\n\n--- USER QUERY ---\nwarmup\n--- END USER QUERY ---"}],"max_tokens":1}' > /tmp/warmup.json && ` +
			"for i in $(seq 1 60); do wget -q -O /dev/null http://localhost:8080/health && break; sleep 1; done && " +
			"wget -q -O /dev/null --post-file=/tmp/warmup.json --header='Content-Type: application/json' http://localhost:8080/v1/chat/completions || true",
	}
}

// llamaCppArgs builds the llama.cpp server arguments based on the model config.
// Context size is sized for the model's actual prompt shape, not output length.
func llamaCppArgs(cfg *config.ExpertConfig, modelPath string, sized config.SizedConfig) []string {
	ctxSize := contextSize(cfg, sized)

	args := []string{
		"--model", modelPath,
		"--host", "0.0.0.0",
		"--port", "8080",
		"--ctx-size", ctxSize,
		"--flash-attn", "on",
		"--repeat-penalty", "1.1",
		"--min-p", "0",
		"--batch-size", "512",
		"--metrics",
		"--mlock",
		"--cache-prompt",
	}
	// For models that support thinking mode (Qwen3, DeepSeek), disable it.
	// The fine-tuned model produces output directly without needing to think.
	// Ensure max_tokens in the inference request is high enough for the full response.
	base := strings.ToLower(cfg.Model.Base)
	if strings.Contains(base, "qwen3") || strings.Contains(base, "deepseek") {
		args = append(args, "--reasoning-budget", "0")
	}
	return args
}

// contextSize picks the llama.cpp --ctx-size for the served model.
//
// Free-form auditors are RAG models: the prompt is dominated by retrieved
// REFERENCE DOCUMENTATION (top-k doc chunks) + any live tool evidence + the
// user's pasted manifest — all INPUT that far exceeds the generated output.
// Sizing the window from output statistics (the old max_output*3 heuristic)
// produced a ~1.3k-token window that could not hold more than ~2 doc chunks and
// returned context-overflow errors on anything larger, silently starving the
// model of the very grounding RAG retrieved. So free-form gets a window sized
// for that input budget; only genuinely long outputs push it higher.
//
// Non-free-form generation (short, structured outputs) keeps the lean
// output-driven sizing.
func contextSize(cfg *config.ExpertConfig, sized config.SizedConfig) string {
	if cfg.Project.IsFreeForm() {
		// Input budget: top-k reference chunks (~500 tok each) + tool evidence +
		// a pasted manifest, plus output headroom. 8192 comfortably fits the
		// high-confidence (5-doc) and broaden (7-doc) retrieval paths; Qwen3-8B
		// supports far larger, so this is conservative.
		const freeFormFloor = 8192
		ctx := freeFormFloor
		if hi := sized.MaxOutputTokens * 4; hi > ctx {
			ctx = hi // unusually long outputs (rare) get even more room
		}
		return fmt.Sprintf("%d", ctx)
	}
	if sized.MaxOutputTokens > 0 {
		ctx := sized.MaxOutputTokens * 3 // short structured output + input + headroom
		if ctx < 512 {
			ctx = 512
		}
		return fmt.Sprintf("%d", ctx)
	}
	return "512"
}
