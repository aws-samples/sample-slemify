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

// teiImage is the Hugging Face Text Embeddings Inference CPU image, pinned.
// TEI serves a sentence-transformer encoder over an HTTP /embed endpoint that
// accepts {"inputs": <str|list[str]>} and returns embedding vectors.
const teiImage = "ghcr.io/huggingface/text-embeddings-inference:cpu-1.5"

// EncoderDeploymentName returns the Deployment/Service name for a project's
// managed encoder.
func EncoderDeploymentName(project string) string {
	return fmt.Sprintf("%s-encoder", project)
}

// EncoderServiceURL returns the in-cluster URL of the managed encoder service.
func EncoderServiceURL(project, ns string) string {
	return fmt.Sprintf("http://%s-encoder.%s.svc.cluster.local:8080", project, ns)
}

// EncoderManifests holds the managed encoder Deployment and Service.
type EncoderManifests struct {
	Deployment *appsv1.Deployment
	Service    *corev1.Service
}

// GenerateEncoderManifests creates a CPU TEI Deployment + Service that serves
// the configured encoder (model.base) for encoder-head tasks. Both the training
// job and the classifier serving pod embed against this endpoint.
func GenerateEncoderManifests(cfg *config.ExpertConfig, ns string, pc *pipeline.PipelineContext) *EncoderManifests {
	name := EncoderDeploymentName(cfg.Project.Name)
	labels := map[string]string{
		"slemify.io/project":           cfg.Project.Name,
		"slemify.io/stage":             "encoder",
		"app.kubernetes.io/managed-by": "slemify",
	}
	selectorLabels := map[string]string{
		"slemify.io/project": cfg.Project.Name,
		"slemify.io/stage":   "encoder",
	}

	replicas := int32(1)
	automountSA := pc.ServiceAccount != ""
	noEscalation := false
	readOnlyRoot := true
	runAsNonRoot := true
	runAsUser := int64(1000)

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels:    labels,
		},
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
							Name:  "tei",
							Image: teiImage,
							Args: []string{
								"--model-id", cfg.Model.Base,
								"--port", "8080",
							},
							Ports: []corev1.ContainerPort{
								{Name: "http", ContainerPort: 8080, Protocol: corev1.ProtocolTCP},
							},
							Env: []corev1.EnvVar{
								// TEI needs a writable cache dir; /tmp is mounted emptyDir.
								{Name: "HUGGINGFACE_HUB_CACHE", Value: "/tmp/hf-cache"},
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
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/health",
										Port: intstr.FromInt32(8080),
									},
								},
								InitialDelaySeconds: 15,
								PeriodSeconds:       5,
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("2"),
									corev1.ResourceMemory: resource.MustParse("4Gi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse("4Gi"),
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

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Type:     corev1.ServiceTypeClusterIP,
			Selector: selectorLabels,
			Ports: []corev1.ServicePort{
				{Name: "http", Port: 8080, TargetPort: intstr.FromInt32(8080), Protocol: corev1.ProtocolTCP},
			},
		},
	}

	return &EncoderManifests{Deployment: deployment, Service: service}
}
