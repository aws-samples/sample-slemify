// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Package pricing looks up the on-demand list price of one EC2 instance type
// so the report can state what a serving node costs per hour. It is the only
// price the report shows: a node is fixed capacity, and turning it into a
// per-request figure needs a request rate the report does not know.
package pricing

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/pricing"
	"github.com/aws/aws-sdk-go-v2/service/pricing/types"
)

// OnDemandHourlyUSD returns the Linux, shared-tenancy, on-demand hourly price
// of instanceType in region. The Pricing API is only served from a few
// regions; us-east-1 carries every region's prices via the regionCode filter.
func OnDemandHourlyUSD(ctx context.Context, region, instanceType string) (float64, error) {
	if instanceType == "" {
		return 0, fmt.Errorf("instance type is empty")
	}
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	cfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion("us-east-1"))
	if err != nil {
		return 0, fmt.Errorf("loading AWS config: %w", err)
	}
	if region == "" {
		return 0, fmt.Errorf("region is empty")
	}
	client := pricing.NewFromConfig(cfg)
	match := types.FilterTypeTermMatch
	out, err := client.GetProducts(ctx, &pricing.GetProductsInput{
		ServiceCode: aws.String("AmazonEC2"),
		MaxResults:  aws.Int32(10),
		Filters: []types.Filter{
			{Type: match, Field: aws.String("instanceType"), Value: aws.String(instanceType)},
			{Type: match, Field: aws.String("regionCode"), Value: aws.String(region)},
			{Type: match, Field: aws.String("operatingSystem"), Value: aws.String("Linux")},
			{Type: match, Field: aws.String("tenancy"), Value: aws.String("Shared")},
			{Type: match, Field: aws.String("preInstalledSw"), Value: aws.String("NA")},
			{Type: match, Field: aws.String("capacitystatus"), Value: aws.String("Used")},
		},
	})
	if err != nil {
		return 0, fmt.Errorf("pricing GetProducts: %w", err)
	}
	for _, raw := range out.PriceList {
		if price, ok := parseOnDemandPrice(raw); ok {
			return price, nil
		}
	}
	return 0, fmt.Errorf("no on-demand price found for %s in %s", instanceType, region)
}

// parseOnDemandPrice digs the USD hourly figure out of one PriceList entry:
// terms.OnDemand.<sku>.priceDimensions.<rate>.pricePerUnit.USD. Entries whose
// product operation is not plain RunInstances (for example SQL Server bundles)
// are skipped by the caller's filters; here we just take the first USD value.
func parseOnDemandPrice(raw string) (float64, bool) {
	var doc struct {
		Product struct {
			Attributes map[string]string `json:"attributes"`
		} `json:"product"`
		Terms struct {
			OnDemand map[string]struct {
				PriceDimensions map[string]struct {
					Unit         string            `json:"unit"`
					PricePerUnit map[string]string `json:"pricePerUnit"`
				} `json:"priceDimensions"`
			} `json:"OnDemand"`
		} `json:"terms"`
	}
	if err := json.Unmarshal([]byte(raw), &doc); err != nil {
		return 0, false
	}
	if op := doc.Product.Attributes["operation"]; op != "" && op != "RunInstances" {
		return 0, false
	}
	for _, term := range doc.Terms.OnDemand {
		for _, dim := range term.PriceDimensions {
			if dim.Unit != "Hrs" {
				continue
			}
			if usd, ok := dim.PricePerUnit["USD"]; ok {
				if v, err := strconv.ParseFloat(usd, 64); err == nil && v > 0 {
					return v, true
				}
			}
		}
	}
	return 0, false
}
