# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""lib/prep.sh with a fake aws, stub crane and a fake corpus export."""
import json
import os
import unittest

from tests._stubs import AWS_FAKE, COMMIT, CRANE_STUB, INFRA, PREFIX, REPOS, TAG, StubEnv, load_buildspec

PREP = os.path.join(INFRA, "lib", "prep.sh")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(INFRA))))
BUCKET = "central-prep-artifacts"
GGUF_KEY = PREFIX + "models/k8s-autoscaling-analyst/model-q4_k_m.gguf"

# Stands in for demo/scripts/index-knowledge.py --export-only: writes three
# corpus lines to the fake S3.
EXPORT = r'''
import os, sys
args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if "=" in a)
assert "--export-only" in sys.argv and "--strict" in sys.argv
path = os.path.join(os.environ["STUB_DIR"], "s3", args["s3-bucket"], args["s3-key"])
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    f.write('{"text": "a", "source": "s", "section": "x"}\n' * 3)
with open(os.environ["STUB_LOG"], "a") as log:
    log.write("export " + args["s3-key"] + "\n")
'''


# Stands in for src/containers/data-pipeline/main.py --pairs-only. It fails the
# first $PAIRS_FAILS runs (a counter file), writes three pairs and a manifest,
# and logs its arguments and the Bedrock regions it was given.
PAIRS_ONLY = r'''
import hashlib, json, os, sys
args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
with open(os.environ["STUB_LOG"], "a") as log:
    log.write("pairs-only %s BEDROCK_REGION=%s BEDROCK_FALLBACK_REGION=%s\n" % (
        " ".join(sys.argv[1:2]), os.environ.get("BEDROCK_REGION"), os.environ.get("BEDROCK_FALLBACK_REGION")))
counter = os.path.join(os.environ["STUB_DIR"], "pairs-runs")
n = int(open(counter).read()) if os.path.exists(counter) else 0
open(counter, "w").write(str(n + 1))
if n < int(os.environ.get("PAIRS_FAILS", "0")):
    print("2026-11-24 10:00:00 [ERROR] slemify.data-pipeline: Generated 12 of 396 expected pairs. Stopping below 198 pairs.", file=sys.stderr)
    sys.exit(1)
body = ('{"query": "q", "positive": "p", "source": "s"}\n' * 3).encode()
open(args["--out"], "wb").write(body)
json.dump({"version": 1, "project": "k8s-autoscaling-retriever", "pairs": 3, "chunks": 3, "expected": 9,
           "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "corpus_sha256": "e" * 64},
          open(args["--manifest"], "w"))
'''
RETRIEVER = PREFIX + "retriever/"
class PrepShTest(unittest.TestCase):
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub_py("aws", AWS_FAKE)
        self.s.stub("crane", CRANE_STUB)
        # Tool setup: curl fails when $CURL_FAIL is 1, pip3 when $PIP_FAIL is 1.
        self.s.stub("curl", '[ "${CURL_FAIL:-0}" = 1 ] && exit 22; exit 0')
        self.s.stub("sha256sum", "cat >/dev/null; exit 0")
        self.s.stub("tar", "exit 0")
        self.s.stub("pip3", '[ "${PIP_FAIL:-0}" = 1 ] && exit 1; exit 0')
        src = self.s.path("src")
        scripts = os.path.join(src, "examples/k8s-autoscaling/demo/scripts")
        os.makedirs(scripts)
        with open(os.path.join(scripts, "index-knowledge.py"), "w") as f:
            f.write(EXPORT)
        pipeline = os.path.join(src, "src/containers/data-pipeline")
        os.makedirs(pipeline)
        with open(os.path.join(pipeline, "main.py"), "w") as f:
            f.write(PAIRS_ONLY)
        self.env = {
            "RELEASE_TAG": TAG, "CODEBUILD_RESOLVED_SOURCE_VERSION": COMMIT,
            "CODEBUILD_SRC_DIR": src, "CODEBUILD_BUILD_ID": "slemify-prep:42",
            "MODEL_BUCKET": BUCKET, "IMAGES_AMD64_PROJECT": "slemify-prep-images-amd64",
            "IMAGES_ARM64_PROJECT": "slemify-prep-images-arm64", "GGUF_PROJECT": "slemify-prep-gguf",
        }
        prep_vars = load_buildspec("buildspec-prep.yaml")["env"]["variables"]
        self.env.update({k: prep_vars[k] for k in ("CRANE_VERSION", "CRANE_SHA256")})

    def prep(self, **extra):
        env = dict(self.env)
        env.update(extra)
        return self.s.run('bash "%s"' % PREP, redirect_tmp=False, timeout=120, **env)

    def upload_index(self, name):
        calls = self.s.calls()
        hits = [i for i, c in enumerate(calls)
                if c.startswith("aws s3 cp") and c.endswith("s3://%s/%s%s" % (BUCKET, PREFIX, name))]
        self.assertEqual(len(hits), 1, name)
        return hits[0]

    def test_skips_when_ready(self):
        self.s.s3_put(BUCKET, PREFIX + "ready.json", "{}")
        r = self.prep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("already prepared", r.stdout)
        self.assertEqual(self.s.calls("aws codebuild start-build"), [])

    def test_full_run_writes_markers_in_order(self):
        r = self.prep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        starts = self.s.calls("aws codebuild start-build")
        self.assertEqual(len(starts), 3)
        self.assertIn("--source-version " + COMMIT, starts[0])
        self.assertIn("name=IMAGE_TAG,value=%s,type=PLAINTEXT" % COMMIT[:12], starts[0])
        self.assertIn("name=GGUF_IMAGE_TAG,value=%s-amd64,type=PLAINTEXT" % COMMIT[:12], starts[2])
        self.assertIn("name=KEY_PREFIX,value=%s,type=PLAINTEXT" % PREFIX, starts[2])
        self.assertEqual(len(self.s.calls("crane index append")), 7)
        self.assertIn("crane copy ghcr.io/ggml-org/llama.cpp@sha256:10a38e6c", self.s.calls("crane copy")[0])
        self.assertLess(self.upload_index("images-ready.json"), self.upload_index("ready.json"))
        images = json.loads(self.s.s3_get(BUCKET, PREFIX + "images-ready.json"))
        self.assertEqual(sorted(images["images"]), sorted(REPOS))
        self.assertEqual(images["knowledge"]["chunks"], 3)
        self.assertEqual(images["commit"], COMMIT)
        ready = json.loads(self.s.s3_get(BUCKET, PREFIX + "ready.json"))
        self.assertEqual(ready["gguf"], {"key": GGUF_KEY, "bytes": 4})
        self.assertEqual(ready["buildId"], "slemify-prep:42")
        with open(os.path.join(self.s.tmp, "prep-step")) as f:
            self.assertEqual(f.read().strip(), "ready")

    def test_gguf_guard_waits_for_running_build(self):
        self.s.state("gguf-builds", "slemify-prep-gguf:old")
        self.s.state("inprogress-slemify-prep-gguf_old", "2")
        r = self.prep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = self.s.calls()
        polls = [i for i, c in enumerate(calls) if "batch-get-builds --ids slemify-prep-gguf:old" in c]
        start = [i for i, c in enumerate(calls) if c.startswith("aws codebuild start-build --project-name slemify-prep-gguf")]
        self.assertEqual(len(polls), 3)
        self.assertEqual(len(start), 1)
        self.assertGreater(start[0], polls[-1])

    def test_gguf_guard_retries_status_reads(self):
        self.s.state("gguf-builds", "slemify-prep-gguf:old")
        self.s.state("fail-list-builds-for-project", "2")
        self.s.state("fail-get-slemify-prep-gguf_old", "1")
        r = self.prep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.s.calls("aws codebuild list-builds-for-project")), 3)
        self.assertEqual(len(self.s.calls("aws codebuild batch-get-builds --ids slemify-prep-gguf:old")), 2)
        self.assertEqual(len(self.s.calls("aws codebuild start-build --project-name slemify-prep-gguf")), 1)

    def test_gguf_guard_fails_closed_when_status_is_unavailable(self):
        for name in ("fail-list-builds-for-project", "fail-get-slemify-prep-gguf_old"):
            with self.subTest(name=name):
                self.s.state("gguf-builds", "slemify-prep-gguf:old")
                self.s.state(name, "always")
                r = self.prep()
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertEqual(self.s.calls("aws codebuild start-build --project-name slemify-prep-gguf"), [])
                self.assertIsNone(self.s.s3_get(BUCKET, PREFIX + "ready.json"))
                self.assertIn("status could not be read; not starting another build", r.stdout)
                with open(os.path.join(self.s.tmp, "prep-step")) as f:
                    self.assertEqual(f.read().strip(), "wait-gguf")
                os.remove(self.s.path("state", name))

    def test_child_failure_fails_and_writes_no_marker(self):
        self.s.state("status-slemify-prep-images-arm64_1", "FAILED")
        r = self.prep()
        self.assertEqual(r.returncode, 1)
        self.assertIsNone(self.s.s3_get(BUCKET, PREFIX + "images-ready.json"))
        with open(os.path.join(self.s.tmp, "prep-step")) as f:
            self.assertEqual(f.read().strip(), "wait-arm64")

    def assert_setup_failure_marker(self, **fail):
        r = self.prep(**fail)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.s.calls("aws codebuild start-build"), [])
        # post_build runs after a failed build phase and writes failed.json.
        cmd = load_buildspec("buildspec-prep.yaml")["phases"]["post_build"]["commands"][0]
        env = dict(self.env, CODEBUILD_BUILD_SUCCEEDING=0, CODEBUILD_BUILD_URL="https://example.com/build",
                   CODEBUILD_SRC_DIR=REPO, INFRA_DIR="examples/k8s-autoscaling/workshop/infrastructure")
        r = self.s.run(cmd, **env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        failed = json.loads(self.s.s3_get(BUCKET, PREFIX + "failed.json"))
        self.assertEqual(failed["step"], "setup")
        self.assertEqual(failed["logUrl"], "https://example.com/build")

    def test_crane_setup_failure_writes_failed_json(self):
        self.assert_setup_failure_marker(CURL_FAIL=1)
        self.assertEqual(self.s.calls("pip3"), [])

    def test_pip_setup_failure_writes_failed_json(self):
        self.assert_setup_failure_marker(PIP_FAIL=1)

    # retriever pairs
    def test_pairs_are_written_and_the_marker_goes_last(self):
        r = self.prep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("wrote %sretriever-ready.json" % RETRIEVER, r.stdout)
        self.assertEqual(self.s.s3_get(BUCKET, RETRIEVER + "train.jsonl").count('"query"'), 3)
        manifest = json.loads(self.s.s3_get(BUCKET, RETRIEVER + "manifest.json"))
        marker = json.loads(self.s.s3_get(BUCKET, RETRIEVER + "retriever-ready.json"))
        self.assertEqual(marker, {"version": 1, "tag": TAG, "commit": COMMIT, "retriever": {
            "key": RETRIEVER + "train.jsonl", "bytes": manifest["bytes"], "pairs": 3,
            "sha256": manifest["sha256"]}})
        self.assertLess(self.upload_index("retriever/train.jsonl"), self.upload_index("retriever/manifest.json"))
        self.assertLess(self.upload_index("retriever/manifest.json"), self.upload_index("retriever/retriever-ready.json"))
        self.assertIsNone(self.s.s3_get(BUCKET, RETRIEVER + "failed.json"))
        self.assertIsNotNone(self.s.s3_get(BUCKET, PREFIX + "ready.json"))
    def test_pairs_run_with_the_regions_and_start_after_the_corpus(self):
        r = self.prep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = self.s.calls()
        pairs = [i for i, c in enumerate(calls) if c.startswith("pairs-only")]
        export = [i for i, c in enumerate(calls) if c.startswith("export")]
        self.assertEqual(len(pairs), 1)
        self.assertGreater(pairs[0], export[0])
        self.assertEqual(calls[pairs[0]],
                         "pairs-only --pairs-only BEDROCK_REGION=us-west-2 BEDROCK_FALLBACK_REGION=us-east-1")
        self.assertTrue(any(c.startswith("pip3 install") and "pyyaml" in c for c in self.s.calls("pip3")))
    def test_the_bedrock_region_can_be_set(self):
        self.prep(BEDROCK_REGION="us-east-1")
        self.assertIn("BEDROCK_REGION=us-east-1 BEDROCK_FALLBACK_REGION=us-east-1", self.s.calls("pairs-only")[0])
    def test_a_failed_pairs_step_does_not_fail_prep(self):
        r = self.prep(PAIRS_FAILS=9)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.s.calls("pairs-only")), 3)
        self.assertIsNone(self.s.s3_get(BUCKET, RETRIEVER + "retriever-ready.json"))
        failed = json.loads(self.s.s3_get(BUCKET, RETRIEVER + "failed.json"))
        self.assertEqual((failed["version"], failed["step"], failed["tag"], failed["commit"]),
                         (1, "retriever-pairs", TAG, COMMIT))
        self.assertIn("Stopping below 198 pairs", failed["reason"])
        self.assertIn("WARN retriever pairs not prepared", r.stdout)
        self.assertIn("teams generate their own", r.stdout)
        self.assertIsNotNone(self.s.s3_get(BUCKET, PREFIX + "images-ready.json"))
        self.assertIsNotNone(self.s.s3_get(BUCKET, PREFIX + "ready.json"))
        self.assertIsNone(self.s.s3_get(BUCKET, PREFIX + "failed.json"))
    def test_a_pairs_failure_is_retried(self):
        r = self.prep(PAIRS_FAILS=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.s.calls("pairs-only")), 2)
        self.assertIsNotNone(self.s.s3_get(BUCKET, RETRIEVER + "retriever-ready.json"))
        self.assertIsNone(self.s.s3_get(BUCKET, RETRIEVER + "failed.json"))
    def test_a_pip_failure_for_the_pairs_does_not_fail_prep(self):
        self.s.stub("pip3", 'case "$*" in *pyyaml*) exit 1;; esac; exit 0')
        r = self.prep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.s.calls("pairs-only"), [])
        failed = json.loads(self.s.s3_get(BUCKET, RETRIEVER + "failed.json"))
        self.assertEqual(failed["step"], "retriever-pairs")
        self.assertIsNotNone(self.s.s3_get(BUCKET, PREFIX + "ready.json"))
    def test_existing_pairs_are_skipped_unless_forced(self):
        self.s.s3_put(BUCKET, RETRIEVER + "retriever-ready.json", "{}")
        r = self.prep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("retriever pairs exist; skipping", r.stdout)
        self.assertEqual(self.s.calls("pairs-only"), [])
        self.assertEqual(self.s.s3_get(BUCKET, RETRIEVER + "retriever-ready.json"), "{}")
        r = self.prep(FORCE=1)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.s.calls("pairs-only")), 1)
        self.assertNotEqual(self.s.s3_get(BUCKET, RETRIEVER + "retriever-ready.json"), "{}")
    def test_a_stale_failed_json_is_removed_when_the_pairs_work(self):
        self.s.s3_put(BUCKET, RETRIEVER + "failed.json", "{}")
        r = self.prep()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIsNone(self.s.s3_get(BUCKET, RETRIEVER + "failed.json"))
        self.assertIsNotNone(self.s.s3_get(BUCKET, RETRIEVER + "retriever-ready.json"))
    def test_invalid_tag(self):
        r = self.prep(RELEASE_TAG="bad/tag")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(self.s.calls("aws codebuild"), [])


if __name__ == "__main__":
    unittest.main()
