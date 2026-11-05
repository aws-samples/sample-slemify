// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package data

import "testing"

func rec(input, output string) Record {
	return Record{"input": input, "output": output}
}

func TestSampleByLabelCoversLabels(t *testing.T) {
	// The committed triage set starts with a long run of one label.
	var recs []Record
	for i := 0; i < 20; i++ {
		recs = append(recs, rec("k", "karpenter_config"))
	}
	recs = append(recs, rec("h", "hpa_tuning"), rec("p", "pdb_disruption | extra"), rec("m", "multi_resource"))

	got := SampleByLabel(recs, 3)
	if len(got) != 3 {
		t.Fatalf("got %d samples, want 3", len(got))
	}
	seen := map[string]bool{}
	for _, r := range got {
		seen[PrimaryLabel(r["output"])] = true
	}
	if len(seen) != 3 {
		t.Errorf("samples cover %d labels, want 3: %v", len(seen), got)
	}
	if PrimaryLabel(got[0]["output"]) != "karpenter_config" || PrimaryLabel(got[1]["output"]) != "hpa_tuning" {
		t.Errorf("labels not in first-seen order: %v", got)
	}
}

func TestSampleByLabelReturnsAllWhenFew(t *testing.T) {
	recs := []Record{rec("a", "x"), rec("b", "x"), rec("c", "y")}
	got := SampleByLabel(recs, 10)
	if len(got) != 3 {
		t.Fatalf("got %d samples, want 3", len(got))
	}
	// Round-robin: x, y, then the second x.
	if got[0]["input"] != "a" || got[1]["input"] != "c" || got[2]["input"] != "b" {
		t.Errorf("unexpected order: %v", got)
	}
}

func TestParseJSONLKeepsStringFields(t *testing.T) {
	data := `{"input": "q1", "output": "a", "score": 3}
not json
{"input": "q2", "output": "b"}
`
	got := ParseJSONL(data)
	if len(got) != 2 {
		t.Fatalf("got %d records, want 2", len(got))
	}
	if got[0]["input"] != "q1" || got[0]["output"] != "a" {
		t.Errorf("record with a numeric field lost its strings: %v", got[0])
	}
	if _, ok := got[0]["score"]; ok {
		t.Errorf("numeric field kept as a string: %v", got[0])
	}
}

func TestPrimaryLabel(t *testing.T) {
	if got := PrimaryLabel("a | b"); got != "a" {
		t.Errorf("PrimaryLabel = %q, want %q", got, "a")
	}
	if got := PrimaryLabel(" plain "); got != "plain" {
		t.Errorf("PrimaryLabel = %q, want %q", got, "plain")
	}
}

func TestCountBySkipsEmpty(t *testing.T) {
	recs := []Record{{"origin": "labeled"}, {"origin": "labeled"}, {"origin": ""}, {}}
	got := CountBy(recs, func(r Record) string { return r["origin"] })
	if len(got) != 1 || got["labeled"] != 2 {
		t.Errorf("CountBy = %v, want map[labeled:2]", got)
	}
}
