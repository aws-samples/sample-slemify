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
def wait_data_function():
    """The wait_data function of seed.sh, as written in the file."""
    with open(SEED) as f:
        text = f.read()
    start = text.index("  wait_data() {")
    return text[start:text.index("\n  }\n", start) + len("\n  }\n")]
class WaitDataTest(unittest.TestCase):
    """A failed data stage prints the failure record the pipeline left in the
    bucket, because the Job controller has deleted the failed pods."""
    def setUp(self):
        from tests._stubs import AWS_FAKE
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub_py("aws", AWS_FAKE)
        self.log = self.s.path("stage.log")
        with open(self.log, "w") as f:
            f.write("[DATA] failed: data job failed\n")
    def run_stage(self, exit_code):
        script = """set -uo pipefail
log() { :; }
SEED_BG=""
%s
sh -c 'exit %d' &
wait_data Retriever $! "$STAGE_LOG" 0 k8s-autoscaling-retriever
echo "after: rc=$?"
""" % (wait_data_function(), exit_code)
        return self.s.run(script, redirect_tmp=False, STAGE_LOG=self.log, MODEL_BUCKET="team-models")
    def test_a_failed_stage_prints_the_failure_record(self):
        record = '{"version": 1, "message": "Generated 12 of 396 expected pairs"}'
        self.s.s3_put("team-models", "k8s-autoscaling-retriever/processed/data-failure.json", record)
        r = self.run_stage(1)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("ERROR: Retriever data stage failed", r.stderr)
        self.assertIn("[DATA] failed: data job failed", r.stderr)
        self.assertIn("Generated 12 of 396 expected pairs", r.stderr)
        self.assertNotIn("after:", r.stdout)
        self.assertEqual(self.s.calls("aws s3 cp"),
                         ["aws s3 cp s3://team-models/k8s-autoscaling-retriever/processed/data-failure.json -"])
    def test_a_failed_stage_without_a_record_still_fails(self):
        r = self.run_stage(1)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("[DATA] failed: data job failed", r.stderr)
    def test_a_stage_that_worked_prints_no_record(self):
        r = self.run_stage(0)
        self.assertIn("after: rc=0", r.stdout)
        self.assertEqual(self.s.calls("aws s3 cp"), [])
class PrebuiltPairsCleanupTest(unittest.TestCase):
    """The central retriever pairs are removed from the model bucket once the
    retriever stage succeeded, and kept when it failed."""
    PREFIX = "k8s-autoscaling/data/prebuilt/k8s-autoscaling-retriever/"
    def setUp(self):
        from tests._stubs import AWS_FAKE
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub_py("aws", AWS_FAKE)
        self.s.s3_put("team-models", self.PREFIX + "train.jsonl", "x")
        self.s.s3_put("team-models", self.PREFIX + "manifest.json", "{}")
        self.s.s3_put("team-models", "k8s-autoscaling/data/queries/doc.txt", "keep")
        with open(self.s.path("stage.log"), "w") as f:
            f.write("log\n")
    def run_seed_tail(self, retriever_exit):
        """wait_data for the retriever, then the cleanup lines that follow it."""
        with open(SEED) as f:
            text = f.read()
        start = text.index("  wait_data() {")
        end = text.index("else\n  echo \"  slemify not installed")
        script = """set -uo pipefail
log() { :; }
SEED_BG=""
TRIAGE_DATA_PID=""
RETRIEVER_DATA_PID=""
fault_bedrock_stop() { :; }
sh -c 'exit %d' &
RETRIEVER_DATA_PID=$!
%s
""" % (retriever_exit, text[start:end])
        script = script.replace('wait_data "Triage" "${TRIAGE_DATA_PID}" /tmp/triage-data.log "${TRIAGE_DATA_T0:-0}" k8s-autoscaling-triage', "")
        script = script.replace("/tmp/retriever-data.log", "$STAGE_LOG")
        return self.s.run(script, redirect_tmp=False, STAGE_LOG=self.s.path("stage.log"), MODEL_BUCKET="team-models")
    def test_removed_after_a_successful_retriever_stage(self):
        r = self.run_seed_tail(0)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIsNone(self.s.s3_get("team-models", self.PREFIX + "train.jsonl"))
        self.assertIsNone(self.s.s3_get("team-models", self.PREFIX + "manifest.json"))
        self.assertEqual(self.s.s3_get("team-models", "k8s-autoscaling/data/queries/doc.txt"), "keep")
        self.assertEqual(self.s.calls("aws s3 rm"),
                         ["aws s3 rm --recursive --quiet s3://team-models/" + self.PREFIX])
    def test_kept_after_a_failed_retriever_stage(self):
        r = self.run_seed_tail(1)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertEqual(self.s.s3_get("team-models", self.PREFIX + "train.jsonl"), "x")
        self.assertEqual(self.s.calls("aws s3 rm"), [])
    def test_a_failed_delete_only_warns(self):
        self.s.stub("aws", 'case "$2" in rm) exit 1;; esac; exit 0')
        r = self.run_seed_tail(0)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("WARN could not remove the prebuilt retriever pairs", r.stderr)
if __name__ == "__main__":
    unittest.main()
