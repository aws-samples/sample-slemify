// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package report

import "testing"

const classificationJSON = `{"project":"p","task":"classification","generated_at":"2026-09-11T00:00:00Z",
"metrics":{"accuracy":0.85,"correct":17,"total":20,"baseline":{"kind":"majority_class","label":"a","accuracy":0.45},
"by_origin":{"real":{"n":8,"correct":5,"accuracy":0.625},"synthetic":{"n":12,"correct":12,"accuracy":1.0}},
"confusions":[{"expected":"b","predicted":"a","count":2}]},
"latency":{"n":30,"p50_ms":12.3,"p95_ms":18.0,"max_ms":31.0},
"llm_baseline":{"model":"m","n":20,"accuracy":0.9},
"serving":{"instance_type":"c8g.2xlarge","vcpus":8,"hourly_usd":0.3547},
"findings":["first thing"],"guidance":["first thing","generic"]}`

func TestParseSummaryClassification(t *testing.T) {
	s, err := ParseSummary(classificationJSON)
	if err != nil {
		t.Fatal(err)
	}
	if s.Task != "classification" || pctStr(s.Metrics, "accuracy") != "85.0%" {
		t.Fatalf("unexpected parse: %+v", s)
	}
	if pctStr(sub(s.Metrics, "baseline"), "accuracy") != "45.0%" {
		t.Fatal("baseline accuracy not parsed")
	}
	if msStr(s.Latency, "p50_ms") != "12 ms" {
		t.Fatalf("latency: %s", msStr(s.Latency, "p50_ms"))
	}
	PrintSummary(s) // must not panic
}

func TestParseSummaryGeneration(t *testing.T) {
	raw := `{"project":"g","task":"generation","profile":{"model_path":"m.gguf","model_size_bytes":1929902912,
"cold":{"ttft_ms":4000,"prompt_tokens":900},"warm":{"ttft_ms":50,"prompt_tokens":900,"decode_tok_s":20.5},
"ceiling":{"tokens_per_second_ceiling":23.2,"bandwidth_gb_s":44.8,"bytes_per_token_assumed":1929902912,"measured_fraction":0.86}},
"grounded_eval":{"pass_rate":0.75,"cases":4,"repeat":2},"serving":{},"guidance":[]}`
	s, err := ParseSummary(raw)
	if err != nil {
		t.Fatal(err)
	}
	PrintSummary(s)
	if s.Grounded == nil || pctStr(s.Grounded, "pass_rate") != "75.0%" {
		t.Fatal("grounded eval not parsed")
	}
}

func TestParseSummaryRejectsGarbage(t *testing.T) {
	if _, err := ParseSummary("nope"); err == nil {
		t.Fatal("expected error")
	}
}
