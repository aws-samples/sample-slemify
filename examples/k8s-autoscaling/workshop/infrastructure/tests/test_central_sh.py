# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""lib/central.sh with a fake aws and stub crane, terraform, timeout and sleep."""
import json
import os
import time
import unittest

from tests._stubs import (AWS_FAKE, COMMIT, CRANE_STUB, INFRA, PREFIX, REPOS, TAG,
                          TERRAFORM_STUB, StubEnv)

LIB = os.path.join(INFRA, "lib", "central.sh")
PARAM = {"version": 1, "account": "111122223333", "region": "us-west-2",
         "bucket": "central-prep-artifacts",
         "registry": "111122223333.dkr.ecr.us-west-2.amazonaws.com/slemify"}
CORPUS = '{"text": "a", "source": "s", "section": "x"}\n' * 3


def images_marker(**over):
    m = {"version": 1, "tag": TAG, "commit": COMMIT,
         "images": {r: "sha256:" + "c" * 64 for r in REPOS},
         "knowledge": {"key": PREFIX + "knowledge/corpus.jsonl", "bytes": len(CORPUS), "chunks": 3}}
    m.update(over)
    return json.dumps(m)


class CentralShTest(unittest.TestCase):
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub_py("aws", AWS_FAKE)
        self.s.stub("crane", CRANE_STUB)
        self.s.stub("terraform", TERRAFORM_STUB)
        self.s.stub("failcmd", 'n=$(cat "$STUB_DIR/fails" 2>/dev/null || echo 0)\n'
                    '[ "$n" -le 0 ] && exit 0\necho $((n - 1)) > "$STUB_DIR/fails"\nexit 1')
        self.base = {"CENTRAL_PARAM": "/slemify/central", "RELEASE_TAG": TAG,
                     "CODEBUILD_RESOLVED_SOURCE_VERSION": COMMIT}

    def sh(self, body, **env):
        e = dict(self.base)
        e.update(env)
        return self.s.run('. "%s"\n%s' % (LIB, body), redirect_tmp=False, **e)

    def tmpfile(self, name):
        path = os.path.join(self.s.tmp, name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return f.read()

    def central_state(self, t0=None):
        with open(os.path.join(self.s.tmp, "central.env"), "w") as f:
            for k in ("account", "region", "bucket", "registry"):
                f.write("export CENTRAL_%s=%s\n" % (k.upper(), PARAM[k]))
        with open(os.path.join(self.s.tmp, "build-t0"), "w") as f:
            f.write(str(int(t0 if t0 is not None else time.time())))

    # retry

    def test_retry_stops_at_attempt_count(self):
        with open(self.s.path("fails"), "w") as f:
            f.write("99")
        r = self.sh("retry 3 0 failcmd x")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(len(self.s.calls("failcmd")), 3)
        self.assertEqual(len(self.s.calls("sleep")), 2)

    def test_retry_succeeds_after_a_failure(self):
        with open(self.s.path("fails"), "w") as f:
            f.write("1")
        r = self.sh("retry 5 0 failcmd x")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.s.calls("failcmd")), 2)

    def test_retry_stops_at_deadline(self):
        # Pauses of 30+j and 60+j fit in 100 s; 120+j does not.
        with open(self.s.path("fails"), "w") as f:
            f.write("99")
        r = self.sh('retry 10 "$(( $(date +%s) + 100 ))" failcmd x')
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(len(self.s.calls("failcmd")), 3)
        self.assertIn("deadline", r.stderr)

    # read_central_param and decide_mode

    def test_read_found(self):
        self.s.state("param", json.dumps(PARAM))
        r = self.sh("read_central_param")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        env = self.tmpfile("central.env")
        self.assertIn("export CENTRAL_BUCKET=central-prep-artifacts", env)
        self.assertIn("export CENTRAL_REGISTRY=111122223333.dkr.ecr.us-west-2.amazonaws.com/slemify", env)

    def test_read_not_found_and_empty_name(self):
        self.assertEqual(self.sh("read_central_param").returncode, 2)
        self.assertEqual(self.sh("read_central_param", CENTRAL_PARAM="").returncode, 2)

    def test_read_other_error(self):
        self.s.state("param-error", "An error occurred (AccessDeniedException)")
        self.assertEqual(self.sh("read_central_param").returncode, 1)

    def test_read_invalid(self):
        for bad in (dict(PARAM, version=2), dict(PARAM, account="12345"),
                    dict(PARAM, region="mars"), dict(PARAM, bucket="Bad_Bucket"),
                    dict(PARAM, registry=PARAM["registry"] + "-b")):
            self.s.state("param", json.dumps(bad))
            r = self.sh("read_central_param")
            self.assertEqual(r.returncode, 3, bad)
        self.s.state("param", "not json")
        self.assertEqual(self.sh("read_central_param").returncode, 3)

    def test_decide_central(self):
        self.s.state("param", json.dumps(dict(PARAM, region="us-east-1",
                     registry="111122223333.dkr.ecr.us-east-1.amazonaws.com/slemify")))
        r = self.sh("decide_mode")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.tmpfile("central-mode").strip(), "central")
        self.assertTrue(self.tmpfile("build-t0").strip().isdigit())
        self.assertIn("WARN", r.stdout)
        self.assertEqual(self.s.calls("aws ssm delete-parameter"), [])

    def test_decide_today_deletes_mirror(self):
        r = self.sh("decide_mode")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.tmpfile("central-mode").strip(), "today")
        self.assertEqual(self.s.calls("aws ssm delete-parameter"),
                         ["aws ssm delete-parameter --name /slemify/mirror"])

    def test_decide_invalid_fails(self):
        self.s.state("param", json.dumps(dict(PARAM, version=0)))
        r = self.sh("decide_mode")
        self.assertEqual(r.returncode, 1)
        self.assertIn("/slemify/central", r.stdout)

    def test_missing_bucket_fails_at_once(self):
        self.s.state("param", json.dumps(PARAM))
        self.s.state("head-bucket-error", "An error occurred (404) when calling the HeadBucket operation: Not Found")
        r = self.sh("decide_mode")
        self.assertEqual(r.returncode, 1)
        self.assertIn("stale", r.stdout)

    def test_forbidden_bucket_continues(self):
        self.s.state("param", json.dumps(PARAM))
        self.s.state("head-bucket-error", "An error occurred (403) when calling the HeadBucket operation: Forbidden")
        r = self.sh("decide_mode")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(self.tmpfile("central-mode").strip(), "central")

    # check_marker and wait_for_object

    def marker_rc(self, text, kind="images"):
        path = os.path.join(self.s.tmp, "m.json")
        with open(path, "w") as f:
            f.write(text)
        return self.sh('check_marker "%s" %s' % (path, kind)).returncode

    def test_check_marker(self):
        self.assertEqual(self.marker_rc(images_marker()), 0)
        self.assertEqual(self.marker_rc(images_marker(commit="f" * 40)), 1)
        self.assertEqual(self.marker_rc(images_marker(tag="v0.2.0")), 1)
        self.assertEqual(self.marker_rc(images_marker(images={r: "sha256:" + "c" * 64 for r in REPOS[:7]})), 1)
        self.assertEqual(self.marker_rc(images_marker(knowledge={"bytes": 0})), 1)
        ready = {"version": 1, "tag": TAG, "commit": COMMIT, "gguf": {"bytes": 5}}
        self.assertEqual(self.marker_rc(json.dumps(ready), "ready"), 0)
        ready["gguf"]["bytes"] = 0
        self.assertEqual(self.marker_rc(json.dumps(ready), "ready"), 1)

    def test_wait_gives_up_at_deadline_and_shows_failed_step(self):
        self.s.s3_put(PARAM["bucket"], PREFIX + "images-ready.json", images_marker(commit="f" * 40))
        self.s.s3_put(PARAM["bucket"], PREFIX + "failed.json",
                      json.dumps({"step": "corpus", "logUrl": "https://example.com/log"}))
        r = self.sh('wait_for_object "s3://%s/%simages-ready.json" "$(( $(date +%%s) + 10 ))" marker_images'
                    % (PARAM["bucket"], PREFIX))
        self.assertEqual(r.returncode, 1)
        self.assertIn("step corpus", r.stdout)

    # central_copy_phase

    def prepare_copy(self):
        self.central_state()
        self.s.s3_put(PARAM["bucket"], PREFIX + "images-ready.json", images_marker())
        self.s.s3_put(PARAM["bucket"], PREFIX + "knowledge/corpus.jsonl", CORPUS)

    def test_copy_phase_success(self):
        self.prepare_copy()
        r = self.sh("central_copy_phase")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        copies = self.s.calls("crane copy")
        self.assertEqual(len(copies), 8)
        self.assertIn("crane copy 111122223333.dkr.ecr.us-west-2.amazonaws.com/slemify/llama-cpp@sha256:"
                      + "c" * 64 + " 222233334444.dkr.ecr.us-west-2.amazonaws.com/slemify/llama-cpp:server-b12063",
                      copies)
        self.assertEqual(self.s.s3_get("team-models", "knowledge/corpus.jsonl"), CORPUS)
        env = self.tmpfile("images.env")
        self.assertIn("DEMO_IMAGE=222233334444.dkr.ecr.us-west-2.amazonaws.com/slemify/k8s-autoscaling-orchestrator:latest", env)
        self.assertIn("KNOWLEDGE_CORPUS=s3://team-models/knowledge/corpus.jsonl", env)
        calls = self.s.calls()
        put = [i for i, c in enumerate(calls) if c.startswith("aws ssm put-parameter --name /slemify/mirror")]
        self.assertEqual(len(put), 1)
        self.assertGreater(put[0], max(i for i, c in enumerate(calls) if c.startswith("crane copy")))
        self.assertIn("--value 222233334444.dkr.ecr.us-west-2.amazonaws.com/slemify", calls[put[0]])

    def test_copy_phase_one_failure(self):
        self.prepare_copy()
        r = self.sh("central_copy_phase", CRANE_FAIL="classifier-trainer")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("copy failed: slemify/classifier-trainer", r.stdout)
        self.assertEqual(self.s.calls("aws ssm put-parameter"), [])
        self.assertIsNone(self.tmpfile("images.env"))

    def test_copy_phase_corpus_size_mismatch(self):
        self.prepare_copy()
        self.s.s3_put(PARAM["bucket"], PREFIX + "images-ready.json",
                      images_marker(knowledge={"bytes": len(CORPUS) + 1}))
        r = self.sh("central_copy_phase")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("copy failed: corpus", r.stdout)
        self.assertEqual(self.s.calls("aws ssm put-parameter"), [])

    # central_tail

    def test_tail_copies_gguf_and_applies_addon(self):
        self.central_state()
        self.s.s3_put(PARAM["bucket"], PREFIX + "models/k8s-autoscaling-analyst/model-q4_k_m.gguf", "GGUF")
        self.s.s3_put(PARAM["bucket"], PREFIX + "ready.json", json.dumps(
            {"version": 1, "tag": TAG, "commit": COMMIT, "gguf": {"bytes": 4}}))
        r = self.sh("central_tail")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.s.s3_get("team-models", "models/k8s-autoscaling-analyst/model-q4_k_m.gguf"), "GGUF")
        self.assertEqual(len(self.s.calls("terraform apply")), 1)

    def test_bucket_copies_do_not_read_source_tags(self):
        # The central bucket policy grants s3:GetObject only. A multipart copy
        # with the CLI default --copy-props reads the source tags and is
        # denied (the 18.5 GB GGUF in the test event).
        self.central_state()
        self.s.s3_put(PARAM["bucket"], PREFIX + "models/k8s-autoscaling-analyst/model-q4_k_m.gguf", "GGUF")
        self.s.s3_put(PARAM["bucket"], PREFIX + "ready.json", json.dumps(
            {"version": 1, "tag": TAG, "commit": COMMIT, "gguf": {"bytes": 4}}))
        r = self.sh("central_tail")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        copies = [c for c in self.s.calls("aws s3 cp") if c.count("s3://") == 2]
        self.assertTrue(copies)
        for c in copies:
            self.assertIn("--copy-props metadata-directive", c)
            self.assertNotIn("--quiet", c)

    def test_tail_fails_when_addon_fails(self):
        self.central_state()
        self.s.s3_put(PARAM["bucket"], PREFIX + "models/k8s-autoscaling-analyst/model-q4_k_m.gguf", "GGUF")
        self.s.s3_put(PARAM["bucket"], PREFIX + "ready.json", json.dumps(
            {"version": 1, "tag": TAG, "commit": COMMIT, "gguf": {"bytes": 4}}))
        r = self.sh("central_tail", TF_APPLY_FAIL=1)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(len(self.s.calls("terraform apply")), 3)


if __name__ == "__main__":
    unittest.main()
