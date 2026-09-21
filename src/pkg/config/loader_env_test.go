// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package config

import "testing"

func TestBucketEnvOverride(t *testing.T) {
	yaml := []byte(`apiVersion: slemify/v1
project:
  name: p
  task: classification
  domain: d
  labels:
    routing: [a, b]
data:
  bucket: committed-bucket
  path: x/
`)
	cfg, _, err := Parse(yaml)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Data.Bucket != "committed-bucket" {
		t.Fatalf("without env: bucket = %q", cfg.Data.Bucket)
	}
	t.Setenv("SLEMIFY_BUCKET", "env-bucket")
	cfg, _, err = Parse(yaml)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Data.Bucket != "env-bucket" {
		t.Fatalf("with SLEMIFY_BUCKET: bucket = %q, want env-bucket", cfg.Data.Bucket)
	}
}
