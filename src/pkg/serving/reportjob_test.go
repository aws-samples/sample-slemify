// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"testing"

	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

// The report divides bandwidth by the bytes the model reads per token. For a
// mixture-of-experts model that is the active set, and only the base model id
// carries the active parameter count, so the Job has to pass it.
func TestReportJobCarriesTheBaseModelID(t *testing.T) {
	cfg := karpenterConfig()
	job := ReportJobManifest(cfg, "slemify", "http://inference", &pipeline.PipelineContext{}, ReportEnv{
		InstanceType: "c9g.8xlarge", VCPUs: 32, HourlyUSD: 1.5,
	})
	env := map[string]string{}
	for _, e := range job.Spec.Template.Spec.Containers[0].Env {
		env[e.Name] = e.Value
	}
	if env["BASE_MODEL"] != cfg.Model.Base {
		t.Fatalf("BASE_MODEL is %q, want %q", env["BASE_MODEL"], cfg.Model.Base)
	}
	if env["INSTANCE_TYPE"] != "c9g.8xlarge" {
		t.Fatalf("INSTANCE_TYPE is %q", env["INSTANCE_TYPE"])
	}
}
