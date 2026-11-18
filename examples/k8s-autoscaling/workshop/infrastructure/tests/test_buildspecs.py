# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Run buildspec command blocks with stub docker, aws, crane and sleep."""
import os
import unittest

from tests._stubs import StubEnv, find_command, load_buildspec

# docker stub: "docker build" fails for images that contain $FAIL_IMAGE, up to
# $FAIL_TIMES times (a counter file per run); "docker push" fails when
# $PUSH_FAIL is 1.
DOCKER = r'''
case "$1" in
  build)
    img="$3"
    if [ -n "${FAIL_IMAGE:-}" ] && [ "${img#*"$FAIL_IMAGE"}" != "$img" ]; then
      n=$(cat "$STUB_DIR/docker-fails" 2>/dev/null || echo 0)
      if [ "$n" -lt "${FAIL_TIMES:-0}" ]; then
        echo $((n + 1)) > "$STUB_DIR/docker-fails"
        exit 1
      fi
    fi
    exit 0 ;;
  push) [ "${PUSH_FAIL:-0}" = 1 ] && exit 1; exit 0 ;;
esac
exit 0
'''


class ImageBuildTest(unittest.TestCase):
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub("docker", DOCKER)
        src = self.s.path("src")
        for d in ("examples/k8s-autoscaling/demo/agent",
                  "examples/k8s-autoscaling/triage", "src/containers"):
            os.makedirs(os.path.join(src, d))
        open(os.path.join(src, "examples/k8s-autoscaling/triage/expert.yaml"), "w").close()
        spec = load_buildspec("buildspec-images.yaml")
        self.env = dict(spec["env"]["variables"])
        self.env.update(REGISTRY="111122223333.dkr.ecr.us-west-2.amazonaws.com",
                        ARCH="amd64", CODEBUILD_SRC_DIR=src)
        self.cmd = find_command(spec, "build", "build_group()")

    def builds(self, needle=""):
        return [c for c in self.s.calls("docker build") if needle in c]

    def test_one_failure_is_retried(self):
        r = self.s.run(self.cmd, FAIL_IMAGE="classifier-trainer", FAIL_TIMES=1, **self.env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.builds("classifier-trainer")), 2)
        self.assertEqual(len(self.builds()), 8)
        self.assertEqual(len(self.s.calls("sleep 120")), 1)

    def test_three_failures_fail_the_build(self):
        r = self.s.run(self.cmd, FAIL_IMAGE="k8s-autoscaling-reranker", FAIL_TIMES=99, **self.env)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(self.builds("reranker")), 3)
        self.assertEqual([c for c in self.s.calls("docker push") if "reranker" in c], [])
        # The loop stops at the failed image.
        self.assertEqual(self.builds("decision"), [])

    def test_push_failure_fails_the_build(self):
        r = self.s.run(self.cmd, PUSH_FAIL=1, **self.env)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(self.s.calls("docker push")), 1)

    def test_image_tag(self):
        r = self.s.run(self.cmd, IMAGE_TAG="0123456789ab", **self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(all(":0123456789ab-amd64" in c for c in self.builds()))
        r = self.s.run(self.cmd, **self.env)
        self.assertIn(":latest-amd64", self.builds()[-1])


class JoinTest(unittest.TestCase):
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub("aws", r'''
case "$1 $2" in
  "codebuild batch-get-builds") printf 'SUCCEEDED\tSUCCEEDED\n' ;;
  "sts get-caller-identity") echo 111122223333 ;;
  "ecr get-login-password") echo token ;;
esac
''')
        self.s.stub("crane", '[ "$1 $2" = "index append" ] && [ "${CRANE_FAIL:-0}" = 1 ] && exit 1; exit 0')
        spec = load_buildspec("buildspec.yaml")
        self.env = dict(spec["env"]["variables"])
        self.cmd = find_command(spec, "build", "crane index append")
        with open(os.path.join(self.s.tmp, "image-builds"), "w") as f:
            f.write("amd:1 arm:1\n")

    def test_append_failure_fails_the_join(self):
        r = self.s.run(self.cmd, CRANE_FAIL=1, **self.env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("index append failed for slemify/k8s-autoscaling-orchestrator", r.stdout)
        self.assertEqual(len(self.s.calls("crane index append")), 1)

    def test_join_succeeds(self):
        r = self.s.run(self.cmd, **self.env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.s.calls("crane index append")), 7)


if __name__ == "__main__":
    unittest.main()
