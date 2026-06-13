// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

// GenerateClassifierInferenceManifests builds the serving manifests for an
// encoder-head classifier. The serving pod runs the lean ONNX classifier image:
// it downloads encoder.onnx + tokenizer.json + head.json from S3, embeds with
// onnxruntime + tokenizers (no torch), applies the head, and exposes an
// OpenAI-compatible /v1/chat/completions endpoint returning "<label>|<confidence>"
// — a drop-in for the generative triage SLM. No GGUF, no GPU.
func GenerateClassifierInferenceManifests(cfg *config.ExpertConfig, sized config.SizedConfig, ns string, pc *pipeline.PipelineContext) *InferenceManifests {
	labels := map[string]string{
		"slemify.io/project":           cfg.Project.Name,
		"slemify.io/stage":             "serving",
		"app.kubernetes.io/managed-by": "slemify",
	}
	selectorLabels := map[string]string{
		"slemify.io/project": cfg.Project.Name,
		"slemify.io/stage":   "serving",
	}

	name := fmt.Sprintf("%s-inference", cfg.Project.Name)
	replicas := int32(1)
	automountSA := pc.ServiceAccount != ""
	noEscalation := false
	readOnlyRoot := true
	runAsNonRoot := true
	runAsUser := int64(1000)

	cpu := sized.InferenceCPU
	if cpu == "" {
		cpu = "2"
	}
	mem := sized.InferenceMemory
	if mem == "" {
		mem = "4Gi"
	}

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns, Labels: labels},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: selectorLabels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: selectorLabels},
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
					Containers: []corev1.Container{
						{
							Name:  "classifier",
							Image: pc.Image("classifier-serving"),
							Env: []corev1.EnvVar{
								// Downloads encoder.onnx + tokenizer.json + head.json
								// from s3://<bucket>/models/<project>/ at startup.
								// Embedding models have no head.json; TASK tells the
								// pod to skip it and serve vectors.
								{Name: "S3_BUCKET", Value: cfg.Data.Bucket},
								{Name: "PROJECT", Value: cfg.Project.Name},
								{Name: "TASK", Value: cfg.Project.Task},
							},
							Ports: []corev1.ContainerPort{
								{Name: "http", ContainerPort: 8080, Protocol: corev1.ProtocolTCP},
							},
							SecurityContext: &corev1.SecurityContext{
								AllowPrivilegeEscalation: &noEscalation,
								ReadOnlyRootFilesystem:   &readOnlyRoot,
								RunAsNonRoot:             &runAsNonRoot,
								RunAsUser:                &runAsUser,
								Capabilities:             &corev1.Capabilities{Drop: []corev1.Capability{"ALL"}},
							},
							ReadinessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt32(8080)},
								},
								InitialDelaySeconds: 15,
								PeriodSeconds:       5,
							},
							LivenessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt32(8080)},
								},
								InitialDelaySeconds: 60,
								PeriodSeconds:       15,
								FailureThreshold:    5,
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse(cpu),
									corev1.ResourceMemory: resource.MustParse(mem),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse(mem),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "tmp", MountPath: "/tmp"},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name:         "tmp",
							VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
						},
					},
				},
			},
		},
	}

	service := inferenceService(cfg, ns, labels, selectorLabels)
	pdb := inferencePDB(cfg, ns, labels, selectorLabels)

	return &InferenceManifests{
		Deployment:          deployment,
		Service:             service,
		PodDisruptionBudget: pdb,
	}
}
