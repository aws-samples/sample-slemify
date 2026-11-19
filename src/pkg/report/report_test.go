// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package report

import (
	"io"
	"os"
	"strings"
	"testing"
)

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

// capture returns everything PrintSummary writes to stdout.
func capture(t *testing.T, s *Summary) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	saved := os.Stdout
	os.Stdout = w
	PrintSummary(s)
	os.Stdout = saved
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

// A model that reads only its active set per token: the whole file is the
// floor (7 tokens/s), the active set the ceiling (72), and the measured 40.3
// falls between them. The printed line used to call the floor the estimate and
// the measurement 560% of it.
const moeProfileJSON = `{"project":"g","task":"generation","profile":{"model_path":"m.gguf","model_size_bytes":18550000000,
"cold":{"ttft_ms":11161,"prompt_tokens":922},"warm":{"ttft_ms":51,"prompt_tokens":922,"decode_tok_s":40.3},
"ceiling":{"tokens_per_second_ceiling":71.7,"tokens_per_second_floor":7.2,"bandwidth_gb_s":133,
"bytes_per_token_assumed":1855000000,"bytes_per_token_whole_file":18550000000,"measured_fraction":0.562}},
"serving":{},"guidance":[]}`

func TestPrintGenerationShowsTheRangeForAnActiveSetModel(t *testing.T) {
	s, err := ParseSummary(moeProfileJSON)
	if err != nil {
		t.Fatal(err)
	}
	out := capture(t, s)
	for _, want := range []string{
		"Estimate:    7 to 72 tokens/s from bandwidth",
		"18.55 GB per token for the whole file",
		"1.85 GB for the active set",
		"measured decode is 40.3 tokens/s, inside that range",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
	// The range bounds the measured rate. It does not establish which weights
	// the model read, so the line must not say that it does.
	for _, unwanted := range []string{"% of the estimate", "about 7 tokens/s",
		"reads its active set", "shared weights"} {
		if strings.Contains(out, unwanted) {
			t.Fatalf("still prints %q in:\n%s", unwanted, out)
		}
	}
}

// The same model measured above the upper end of its range. The line used to
// fall through to the dense branch and call the range a floor, one line under
// a printed range whose upper end is a ceiling.
func TestPrintGenerationReportsAMeasurementAboveTheRange(t *testing.T) {
	s, err := ParseSummary(`{"project":"g","task":"generation","profile":{"model_path":"m.gguf","model_size_bytes":18550000000,
"warm":{"ttft_ms":51,"decode_tok_s":80},
"ceiling":{"tokens_per_second_ceiling":71.7,"tokens_per_second_floor":7.2,"bandwidth_gb_s":133,
"bytes_per_token_assumed":1855000000,"bytes_per_token_whole_file":18550000000,"measured_fraction":1.116}},
"serving":{},"guidance":[]}`)
	if err != nil {
		t.Fatal(err)
	}
	out := capture(t, s)
	for _, want := range []string{
		"Estimate:    7 to 72 tokens/s from bandwidth",
		"measured decode is 80.0 tokens/s, above that range",
		"vCPU share of the socket's bandwidth",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
	for _, unwanted := range []string{"is a floor", "% of the estimate", "inside that range"} {
		if strings.Contains(out, unwanted) {
			t.Fatalf("still prints %q in:\n%s", unwanted, out)
		}
	}
}

func TestPrintGenerationKeepsOneEstimateForADenseModel(t *testing.T) {
	s, err := ParseSummary(`{"project":"g","task":"generation","profile":{"model_size_bytes":1929902912,
"warm":{"ttft_ms":50,"decode_tok_s":20.5},
"ceiling":{"tokens_per_second_ceiling":23.2,"tokens_per_second_floor":23.2,"bandwidth_gb_s":44.8,
"bytes_per_token_assumed":1929902912,"bytes_per_token_whole_file":1929902912,"measured_fraction":0.86}},
"serving":{},"guidance":[]}`)
	if err != nil {
		t.Fatal(err)
	}
	out := capture(t, s)
	if !strings.Contains(out, "Estimate:    about 23 tokens/s from bandwidth (45 GB/s share / 1.93 GB per token)") {
		t.Fatalf("dense estimate line changed:\n%s", out)
	}
	if !strings.Contains(out, "measured decode is 86% of the estimate: bandwidth-bound") {
		t.Fatalf("dense reading line changed:\n%s", out)
	}
}
