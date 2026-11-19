# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The demo manifest step of seed/seed.sh, run with a stub kubectl.

The re-ranker and decision Deployments must be applied at zero replicas.
At one replica they preempt the 30 CPU slm-prewarm pod, which makes Auto Mode
launch a second SLM node for a few minutes in every team.
"""
import os
import re
import unittest
from tests._stubs import INFRA, StubEnv
SEED = os.path.join(INFRA, "seed", "seed.sh")
MANIFEST = os.path.join(os.path.dirname(os.path.dirname(INFRA)), "demo", "k8s-manifest.yaml")
START = 'manifest="$(mktemp)"'
END = "kubectl scale deployment/k8s-autoscaling-decision"
IMAGES = {
    "REPLACE_WITH_ECR_IMAGE": "demo-image:1",
    "REPLACE_WITH_RERANKER_IMAGE": "reranker-image:1",
    "REPLACE_WITH_DECISION_IMAGE": "decision-image:1",
}
def demo_step():
    """The seed.sh lines from creating the temp manifest to the last scale call."""
    with open(SEED) as f:
        text = f.read()
    start = text.index(START)
    end = text.index(END, start)
    return text[start:text.index("\n", end)]
def split_docs(text):
    return re.split(r"(?m)^---[ \t]*$", text)
def doc_kind_name(doc):
    kind = re.search(r"(?m)^kind: (\S+)", doc)
    name = re.search(r"(?m)^  name: (\S+)", doc)
    return (kind.group(1), name.group(1)) if kind and name else None
def deployment_replicas(text):
    out = {}
    for doc in split_docs(text):
        kn = doc_kind_name(doc)
        if kn and kn[0] == "Deployment":
            out[kn[1]] = int(re.search(r"(?m)^  replicas: (\d+)", doc).group(1))
    return out
class SeedDemoManifestTest(unittest.TestCase):
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.applied = self.s.path("applied.yaml")
        # kubectl apply -f <file> keeps a copy of the file; other calls only log.
        self.s.stub("kubectl", r'''
if [ "$1" = apply ] && [ "$2" = -f ]; then cp "$3" "$STUB_DIR/applied.yaml"; fi
exit 0''')

    def run_step(self):
        script = "set -euo pipefail\n%s\n" % demo_step()
        return self.s.run(
            script, redirect_tmp=False, TMPDIR=self.s.tmp, NAMESPACE="slemify",
            DEMO_DIR=os.path.dirname(MANIFEST), DEMO_IMAGE="demo-image:1",
            RERANKER_IMAGE="reranker-image:1", DECISION_IMAGE="decision-image:1")

    def test_manifest_ships_the_models_at_one_replica(self):
        # The fix lives in seed.sh because the shared manifest must stay at 1.
        with open(MANIFEST) as f:
            replicas = deployment_replicas(f.read())
        self.assertEqual(replicas["k8s-autoscaling-reranker"], 1)
        self.assertEqual(replicas["k8s-autoscaling-decision"], 1)

    def test_models_are_applied_at_zero_replicas(self):
        r = self.run_step()
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self.applied) as f:
            replicas = deployment_replicas(f.read())
        self.assertEqual(replicas["k8s-autoscaling-reranker"], 0)
        self.assertEqual(replicas["k8s-autoscaling-decision"], 0)
        self.assertEqual(replicas["k8s-autoscaling-orchestrator"], 1)
        self.assertEqual(replicas["k8s-autoscaling-tools"], 1)

    def test_only_the_two_replica_lines_change(self):
        r = self.run_step()
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(MANIFEST) as f:
            text = f.read()
        for placeholder, image in IMAGES.items():
            text = text.replace(placeholder, image)
        with open(self.applied) as f:
            applied = f.read()
        before, after = text.splitlines(), applied.splitlines()
        self.assertEqual(len(before), len(after))
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        self.assertEqual(changed, [("  replicas: 1", "  replicas: 0")] * 2)
        self.assertEqual(len(split_docs(text)), len(split_docs(applied)))

    def test_scale_down_guards_stay(self):
        r = self.run_step()
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = self.s.calls("kubectl scale")
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c.endswith("--replicas=0") for c in calls), calls)

    def test_rewrite_fails_when_a_deployment_is_missing(self):
        # A renamed Deployment must stop the seed, not apply the model at 1.
        script = "set -euo pipefail\n%s\n" % demo_step().replace(
            "k8s-autoscaling-decision <<'PY'", "k8s-autoscaling-renamed <<'PY'")
        self.assertNotEqual(script, "set -euo pipefail\n%s\n" % demo_step())
        r = self.s.run(
            script, redirect_tmp=False, TMPDIR=self.s.tmp, NAMESPACE="slemify",
            DEMO_DIR=os.path.dirname(MANIFEST), DEMO_IMAGE="d", RERANKER_IMAGE="r", DECISION_IMAGE="x")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(os.path.exists(self.applied))
if __name__ == "__main__":
    unittest.main()
