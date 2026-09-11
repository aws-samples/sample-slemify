package serving

import (
	"strings"
	"testing"

	"github.com/aws-samples/sample-slemify/pkg/pipeline"
)

func karpenterOpts() NodePoolOptions {
	return NodePoolOptions{Provisioner: ProvisionerKarpenter}
}

func autoModeOpts() NodePoolOptions {
	return NodePoolOptions{Provisioner: ProvisionerAutoMode, NodeClassName: "default"}
}

// pools splits the two-document manifest into (preferred, fallback).
func pools(t *testing.T, opts NodePoolOptions) (string, string) {
	t.Helper()
	docs := pipeline.SplitYAMLDocs(SLMNodePoolManifests(sized7B(), opts))
	if len(docs) != 2 {
		t.Fatalf("expected 2 NodePool documents, got %d", len(docs))
	}
	return docs[0], docs[1]
}

func TestSLMNodePoolsTwoWeightedPools(t *testing.T) {
	for _, opts := range []NodePoolOptions{karpenterOpts(), autoModeOpts()} {
		preferred, fallback := pools(t, opts)
		if !strings.Contains(preferred, "name: slemify-slm\n") || !strings.Contains(preferred, "weight: 100") {
			t.Errorf("%s: preferred pool should be slemify-slm with weight 100", opts.Provisioner)
		}
		if !strings.Contains(fallback, "name: slemify-slm-fallback") || !strings.Contains(fallback, "weight: 50") {
			t.Errorf("%s: fallback pool should be slemify-slm-fallback with weight 50", opts.Provisioner)
		}
	}
}

func TestSLMNodePoolsGenerations(t *testing.T) {
	preferred, fallback := pools(t, karpenterOpts())
	if !strings.Contains(preferred, `values: ["8"]`) {
		t.Error("preferred pool should be generation 8 only")
	}
	if !strings.Contains(fallback, `values: ["6", "7"]`) {
		t.Error("fallback pool should allow generations 6 and 7")
	}
	for _, m := range []string{preferred, fallback} {
		if strings.Contains(m, `"5"`) || strings.Contains(m, "Gt") {
			t.Error("generation 5 and older must not be eligible in any pool")
		}
	}
}

func TestSLMNodePoolsSharedContract(t *testing.T) {
	// Workloads select nodes by label and toleration; that contract must be
	// identical across both pools and both provisioners.
	for _, opts := range []NodePoolOptions{karpenterOpts(), autoModeOpts()} {
		preferred, fallback := pools(t, opts)
		for _, m := range []string{preferred, fallback} {
			for _, want := range []string{
				"slemify.io/workload: slm", "key: slemify.io/slm", "effect: NoSchedule",
				`"on-demand"`, `"arm64"`, `"amd64"`, `"c", "m", "r"`,
				`"metal", "nano", "micro", "small"`, "WhenEmptyOrUnderutilized",
			} {
				if !strings.Contains(m, want) {
					t.Errorf("%s pool missing %q", opts.Provisioner, want)
				}
			}
			if strings.Contains(m, `"spot"`) {
				t.Errorf("%s pool must not include spot", opts.Provisioner)
			}
		}
	}
}

func TestSLMNodePoolsKarpenterReferencesOwnNodeClass(t *testing.T) {
	preferred, fallback := pools(t, karpenterOpts())
	for _, m := range []string{preferred, fallback} {
		if !strings.Contains(m, "group: karpenter.k8s.aws") || !strings.Contains(m, "kind: EC2NodeClass") {
			t.Error("Karpenter pools should reference a karpenter.k8s.aws EC2NodeClass")
		}
		if !strings.Contains(m, "name: slemify-slm\n") {
			t.Error("Karpenter pools should reference the slemify-slm EC2NodeClass")
		}
		if !strings.Contains(m, "karpenter.k8s.aws/instance-category") {
			t.Error("Karpenter pools should use karpenter.k8s.aws instance labels")
		}
		if strings.Contains(m, "eks.amazonaws.com") {
			t.Error("Karpenter pools must not reference Auto Mode groups")
		}
	}
}

func TestSLMNodePoolsAutoModeReferencesClusterNodeClass(t *testing.T) {
	preferred, fallback := pools(t, autoModeOpts())
	for _, m := range []string{preferred, fallback} {
		if !strings.Contains(m, "group: eks.amazonaws.com") || !strings.Contains(m, "kind: NodeClass") {
			t.Error("Auto Mode pools should reference an eks.amazonaws.com NodeClass")
		}
		if !strings.Contains(m, "name: default") {
			t.Error("Auto Mode pools should reference the named cluster NodeClass")
		}
		for _, key := range []string{"eks.amazonaws.com/instance-category", "eks.amazonaws.com/instance-generation", "eks.amazonaws.com/instance-size"} {
			if !strings.Contains(m, key) {
				t.Errorf("Auto Mode pools should use %s", key)
			}
		}
		if strings.Contains(m, "EC2NodeClass") || strings.Contains(m, "karpenter.k8s.aws") {
			t.Error("Auto Mode pools must not reference self-managed Karpenter types or labels")
		}
	}
}

func TestSLMNodePoolsAutoModeDefaultsNodeClass(t *testing.T) {
	preferred, _ := pools(t, NodePoolOptions{Provisioner: ProvisionerAutoMode})
	if !strings.Contains(preferred, "name: default") {
		t.Error("Auto Mode should default to the NodeClass named default")
	}
}

func TestSLMEC2NodeClass(t *testing.T) {
	manifest := SLMEC2NodeClassManifest("my-cluster", "KarpenterNodeRole-my-cluster", "test-project")

	if !strings.Contains(manifest, "name: slemify-slm") {
		t.Error("should be named slemify-slm")
	}
	if !strings.Contains(manifest, "bottlerocket@latest") {
		t.Error("should use Bottlerocket")
	}
	if !strings.Contains(manifest, "karpenter.sh/discovery: my-cluster") {
		t.Error("should use cluster name for discovery")
	}
	if !strings.Contains(manifest, "encrypted: true") {
		t.Error("EBS should be encrypted")
	}
	if !strings.Contains(manifest, "80Gi") {
		t.Error("should have 80Gi root volume for model processing")
	}
	if !strings.Contains(manifest, "KarpenterNodeRole-my-cluster") {
		t.Error("should reference the node role")
	}
}
