// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package pricing

import "testing"

const sample = `{"product":{"attributes":{"instanceType":"c8g.2xlarge","operation":"RunInstances"}},
"terms":{"OnDemand":{"SKU.TERM":{"priceDimensions":{"SKU.TERM.RATE":{"unit":"Hrs","pricePerUnit":{"USD":"0.3547000000"}}}}}}}`

func TestParseOnDemandPrice(t *testing.T) {
	v, ok := parseOnDemandPrice(sample)
	if !ok || v < 0.35 || v > 0.36 {
		t.Fatalf("expected 0.3547, got %v ok=%v", v, ok)
	}
}

func TestParseOnDemandPriceSkipsOtherOperations(t *testing.T) {
	raw := `{"product":{"attributes":{"operation":"RunInstances:0102"}},"terms":{"OnDemand":{"a":{"priceDimensions":{"b":{"unit":"Hrs","pricePerUnit":{"USD":"9"}}}}}}}`
	if _, ok := parseOnDemandPrice(raw); ok {
		t.Fatal("SQL Server and similar operations must be skipped")
	}
}

func TestParseOnDemandPriceRejectsGarbage(t *testing.T) {
	if _, ok := parseOnDemandPrice("not json"); ok {
		t.Fatal("garbage must not parse")
	}
	if _, ok := parseOnDemandPrice(`{"terms":{"OnDemand":{}}}`); ok {
		t.Fatal("empty terms must not parse")
	}
}
