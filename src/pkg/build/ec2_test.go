// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package build

import (
	"encoding/base64"
	"strings"
	"testing"
)

func TestDefaultBuildSpecs(t *testing.T) {
	specs := DefaultBuildSpecs()

	if len(specs) != 2 {
		t.Fatalf("DefaultBuildSpecs() returned %d specs, want 2", len(specs))
	}

	arches := map[string]bool{}
	for _, s := range specs {
		arches[s.Arch] = true
		if s.InstanceType == "" {
			t.Errorf("spec for %s has empty instance type", s.Arch)
		}
	}

	if !arches["amd64"] {
		t.Error("missing amd64 spec")
	}
	if !arches["arm64"] {
		t.Error("missing arm64 spec")
	}
}

func TestDefaultBuildSpecsInstanceTypes(t *testing.T) {
	specs := DefaultBuildSpecs()
	for _, s := range specs {
		switch s.Arch {
		case "amd64":
			if !strings.HasPrefix(s.InstanceType, "c5") {
				t.Errorf("amd64 instance type = %q, expected c5 family", s.InstanceType)
			}
		case "arm64":
			if !strings.HasPrefix(s.InstanceType, "c6g") {
				t.Errorf("arm64 instance type = %q, expected c6g family (Graviton)", s.InstanceType)
			}
		}
	}
}

func TestUserDataScript(t *testing.T) {
	script := UserDataScript()

	if !strings.HasPrefix(script, "#!/bin/bash") {
		t.Error("user-data should start with shebang")
	}
	if !strings.Contains(script, "docker") {
		t.Error("user-data should install docker")
	}
	if !strings.Contains(script, "systemctl start docker") {
		t.Error("user-data should start docker service")
	}
	if !strings.Contains(script, "build-ready") {
		t.Error("user-data should signal readiness")
	}
}

func TestEncodeUserData(t *testing.T) {
	script := "#!/bin/bash\necho hello"
	encoded := EncodeUserData(script)

	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatalf("encoded user-data is not valid base64: %v", err)
	}
	if string(decoded) != script {
		t.Errorf("decoded = %q, want %q", string(decoded), script)
	}
}

func TestBuildInstanceStruct(t *testing.T) {
	inst := &BuildInstance{
		InstanceID: "i-1234567890abcdef0",
		Arch:       "arm64",
	}

	if inst.InstanceID != "i-1234567890abcdef0" {
		t.Errorf("InstanceID = %q", inst.InstanceID)
	}
	if inst.Arch != "arm64" {
		t.Errorf("Arch = %q", inst.Arch)
	}
}
