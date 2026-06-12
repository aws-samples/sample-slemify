// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package build

import (
	"testing"
)

func TestRegistryURL(t *testing.T) {
	m := &ECRManager{
		accountID: "123456789012",
		region:    "us-east-1",
	}

	expected := "123456789012.dkr.ecr.us-east-1.amazonaws.com"
	if m.RegistryURL() != expected {
		t.Errorf("RegistryURL() = %q, want %q", m.RegistryURL(), expected)
	}
}

func TestRegistryURLDifferentRegion(t *testing.T) {
	m := &ECRManager{
		accountID: "987654321098",
		region:    "eu-west-1",
	}

	expected := "987654321098.dkr.ecr.eu-west-1.amazonaws.com"
	if m.RegistryURL() != expected {
		t.Errorf("RegistryURL() = %q, want %q", m.RegistryURL(), expected)
	}
}

func TestImageURI(t *testing.T) {
	m := &ECRManager{
		accountID: "123456789012",
		region:    "us-east-1",
	}

	uri := m.ImageURI("data-pipeline", "latest")
	expected := "123456789012.dkr.ecr.us-east-1.amazonaws.com/slemify/data-pipeline:latest"
	if uri != expected {
		t.Errorf("ImageURI() = %q, want %q", uri, expected)
	}
}

func TestImageURIWithVersion(t *testing.T) {
	m := &ECRManager{
		accountID: "123456789012",
		region:    "us-east-1",
	}

	uri := m.ImageURI("mcp-server", "v0.1.0")
	expected := "123456789012.dkr.ecr.us-east-1.amazonaws.com/slemify/mcp-server:v0.1.0"
	if uri != expected {
		t.Errorf("ImageURI() = %q, want %q", uri, expected)
	}
}

func TestContainerRepos(t *testing.T) {
	repos := ContainerRepos()

	if len(repos) != 3 {
		t.Fatalf("ContainerRepos() returned %d repos, want 3", len(repos))
	}

	expected := map[string]bool{
		"slemify/data-pipeline":      true,
		"slemify/classifier-trainer": true,
		"slemify/classifier-serving": true,
	}

	for _, repo := range repos {
		if !expected[repo] {
			t.Errorf("unexpected repo %q", repo)
		}
	}
}

func TestAccountIDAndRegion(t *testing.T) {
	m := &ECRManager{
		accountID: "111222333444",
		region:    "ap-southeast-1",
	}

	if m.AccountID() != "111222333444" {
		t.Errorf("AccountID() = %q", m.AccountID())
	}
	if m.Region() != "ap-southeast-1" {
		t.Errorf("Region() = %q", m.Region())
	}
}
