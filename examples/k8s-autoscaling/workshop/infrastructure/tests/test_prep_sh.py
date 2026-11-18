# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""lib/prep.sh with a fake aws, stub crane and a fake corpus export."""
import json
import os
import unittest

from tests._stubs import AWS_FAKE, COMMIT, CRANE_STUB, INFRA, PREFIX, REPOS, TAG, StubEnv

PREP = os.path.join(INFRA, "lib", "prep.sh")
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


class PrepShTest(unittest.TestCase):
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub_py("aws", AWS_FAKE)
        self.s.stub("crane", CRANE_STUB)
        src = self.s.path("src")
        scripts = os.path.join(src, "examples/k8s-autoscaling/demo/scripts")
        os.makedirs(scripts)
        with open(os.path.join(scripts, "index-knowledge.py"), "w") as f:
            f.write(EXPORT)
        self.env = {
            "RELEASE_TAG": TAG, "CODEBUILD_RESOLVED_SOURCE_VERSION": COMMIT,
            "CODEBUILD_SRC_DIR": src, "CODEBUILD_BUILD_ID": "slemify-prep:42",
            "MODEL_BUCKET": BUCKET, "IMAGES_AMD64_PROJECT": "slemify-prep-images-amd64",
            "IMAGES_ARM64_PROJECT": "slemify-prep-images-arm64", "GGUF_PROJECT": "slemify-prep-gguf",
        }

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

    def test_child_failure_fails_and_writes_no_marker(self):
        self.s.state("status-slemify-prep-images-arm64_1", "FAILED")
        r = self.prep()
        self.assertEqual(r.returncode, 1)
        self.assertIsNone(self.s.s3_get(BUCKET, PREFIX + "images-ready.json"))
        with open(os.path.join(self.s.tmp, "prep-step")) as f:
            self.assertEqual(f.read().strip(), "wait-arm64")

    def test_invalid_tag(self):
        r = self.prep(RELEASE_TAG="bad/tag")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(self.s.calls("aws codebuild"), [])


if __name__ == "__main__":
    unittest.main()
