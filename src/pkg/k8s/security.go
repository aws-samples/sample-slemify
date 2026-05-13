// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Package k8s provides Kubernetes client utilities and shared security helpers.
package k8s

import (
	corev1 "k8s.io/api/core/v1"
)

// RestrictedSecurityContext returns a hardened SecurityContext suitable for
// most containers. It enforces non-root execution, read-only root filesystem,
// no privilege escalation, and drops all capabilities.
func RestrictedSecurityContext() *corev1.SecurityContext {
	runAsNonRoot := true
	readOnly := true
	noEscalation := false
	return &corev1.SecurityContext{
		RunAsNonRoot:             &runAsNonRoot,
		ReadOnlyRootFilesystem:   &readOnly,
		AllowPrivilegeEscalation: &noEscalation,
		Capabilities: &corev1.Capabilities{
			Drop: []corev1.Capability{"ALL"},
		},
	}
}

// RestrictedPodSecurityContext returns a hardened pod-level SecurityContext.
func RestrictedPodSecurityContext() *corev1.PodSecurityContext {
	runAsNonRoot := true
	return &corev1.PodSecurityContext{
		RunAsNonRoot: &runAsNonRoot,
	}
}
