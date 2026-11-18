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

    def test_invalid_tag(self):
        r = self.prep(RELEASE_TAG="bad/tag")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(self.s.calls("aws codebuild"), [])


if __name__ == "__main__":
    unittest.main()
