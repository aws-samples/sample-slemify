// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"fmt"

	"github.com/aws-samples/sample-slemify/pkg/config"
)

// KEDAScaledObjectManifest generates a KEDA ScaledObject YAML string
// targeting the inference Deployment with a Prometheus trigger on
// llama.cpp's llamacpp_requests_processing metric.
func KEDAScaledObjectManifest(cfg *config.ExpertConfig, sized config.SizedConfig, ns string) string {
	return fmt.Sprintf(`apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: %s-inference
  namespace: %s
  labels:
    slemify.io/project: %s
    slemify.io/stage: serving
    app.kubernetes.io/managed-by: slemify
spec:
  scaleTargetRef:
    name: %s-inference
  minReplicaCount: 1
  maxReplicaCount: %d
  pollingInterval: 15
  cooldownPeriod: 60
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus-server.monitoring.svc.cluster.local
        metricName: llamacpp_requests_processing
        query: sum(llamacpp_requests_processing{deployment="%s-inference"})
        threshold: "5"
`, cfg.Project.Name, ns, cfg.Project.Name, cfg.Project.Name, sized.KEDAMaxReplicas, cfg.Project.Name)
}
