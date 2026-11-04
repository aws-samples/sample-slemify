// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package data

import (
	"encoding/json"
	"strings"
)

// Record is one JSONL training record, keeping its string fields
// (instruction, input, output, origin).
type Record map[string]string

// ParseJSONL parses JSONL text into records. A record keeps its string fields
// even when other fields hold numbers or objects. Lines that are not valid
// JSON objects are skipped.
func ParseJSONL(data string) []Record {
	var records []Record
	for _, line := range strings.Split(data, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var raw map[string]any
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			continue
		}
		r := Record{}
		for k, v := range raw {
			if s, ok := v.(string); ok {
				r[k] = s
			}
		}
		records = append(records, r)
	}
	return records
}

// PrimaryLabel returns the first pipe-delimited field of an output, trimmed.
func PrimaryLabel(output string) string {
	if idx := strings.Index(output, "|"); idx >= 0 {
		output = output[:idx]
	}
	return strings.TrimSpace(output)
}

// CountBy counts records by the key function. Empty keys are not counted.
func CountBy(records []Record, key func(Record) string) map[string]int {
	counts := make(map[string]int)
	for _, r := range records {
		if k := key(r); k != "" {
			counts[k]++
		}
	}
	return counts
}

// SampleByLabel picks up to n records, one label at a time: round-robin over
// the labels in first-seen order, taking each label's records in file order.
// The result is deterministic, and n records cover min(n, labels) labels.
func SampleByLabel(records []Record, n int) []Record {
	var order []string
	byLabel := make(map[string][]Record)
	for _, r := range records {
		l := PrimaryLabel(r["output"])
		if _, ok := byLabel[l]; !ok {
			order = append(order, l)
		}
		byLabel[l] = append(byLabel[l], r)
	}
	var out []Record
	for round := 0; len(out) < n; round++ {
		added := false
		for _, l := range order {
			if len(out) == n {
				break
			}
			if round < len(byLabel[l]) {
				out = append(out, byLabel[l][round])
				added = true
			}
		}
		if !added {
			break
		}
	}
	return out
}
