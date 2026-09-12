// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package k8s

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// NodeInfo describes the node an inference pod landed on. The report uses it
// to name the instance type, estimate its memory bandwidth, and state the
// on-demand rate of one node.
type NodeInfo struct {
	NodeName     string
	InstanceType string
	VCPUs        int
}

// InferenceNodeInfo finds the first running serving pod of a project and
// returns the instance type and vCPU count of its node. Instance type comes
// from the node.kubernetes.io/instance-type label (set by every provisioner,
// including EKS Auto Mode); vCPUs from the node's CPU capacity.
func (c *Client) InferenceNodeInfo(ctx context.Context, project string) (*NodeInfo, error) {
	selector := fmt.Sprintf("slemify.io/project=%s,slemify.io/stage=serving", project)
	pods, err := c.clientset.CoreV1().Pods(c.namespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
	if err != nil {
		return nil, fmt.Errorf("listing serving pods: %w", err)
	}
	var nodeName string
	for _, pod := range pods.Items {
		if pod.Status.Phase == corev1.PodRunning && pod.Spec.NodeName != "" {
			nodeName = pod.Spec.NodeName
			break
		}
	}
	if nodeName == "" {
		return nil, fmt.Errorf("no running serving pod for %s", project)
	}
	node, err := c.clientset.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("getting node %s: %w", nodeName, err)
	}
	info := &NodeInfo{NodeName: nodeName}
	info.InstanceType = node.Labels["node.kubernetes.io/instance-type"]
	if info.InstanceType == "" {
		info.InstanceType = node.Labels["beta.kubernetes.io/instance-type"]
	}
	if cpu, ok := node.Status.Capacity[corev1.ResourceCPU]; ok {
		info.VCPUs = int(cpu.Value())
	}
	return info, nil
}
