# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Run buildspec command blocks with stub docker, aws, crane and sleep."""
import json
import os
import unittest

from tests._stubs import INFRA, StubEnv, find_command, load_buildspec, phase_commands

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


class ModeGuardTest(unittest.TestCase):
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub("aws", "echo build:1")
        self.cmd = find_command(load_buildspec("buildspec.yaml"), "pre_build", "IMAGE_BUILD_AMD64")
        self.env = {"IMAGE_BUILD_AMD64": "a", "IMAGE_BUILD_ARM64": "b"}

    def start(self, mode):
        with open(os.path.join(self.s.tmp, "central-mode"), "w") as f:
            f.write(mode + "\n")
        self.assertEqual(self.s.run(self.cmd, **self.env).returncode, 0)
        return self.s.calls("aws codebuild start-build")

    def test_central_mode_starts_no_image_build(self):
        self.assertEqual(self.start("central"), [])

    def test_today_starts_both(self):
        self.assertEqual(len(self.start("today")), 2)


class ModeFailureTest(unittest.TestCase):
    """A bad /slemify/central must fail in the build phase, where post_build
    still runs and sends FAILURE to the stack, not in install."""

    def setUp(self):
        from tests._stubs import AWS_FAKE
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub_py("aws", AWS_FAKE)
        spec = load_buildspec("buildspec.yaml")
        self.decide = find_command(spec, "install", "decide_mode")
        self.check = phase_commands(spec, "build")[0]
        self.env = dict(spec["env"]["variables"])
        self.env.update(CODEBUILD_SRC_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(INFRA)))),
                        CENTRAL_PARAM="/slemify/central")

    def mode(self):
        with open(os.path.join(self.s.tmp, "central-mode")) as f:
            return f.read().strip()

    def test_invalid_parameter_fails_in_build(self):
        self.s.state("param", json.dumps({"version": 0}))
        r = self.s.run(self.decide, **self.env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.mode(), "failed")
        r = self.s.run(self.check, **self.env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("/slemify/central", r.stdout)

    def test_today_passes_the_build_check(self):
        r = self.s.run(self.decide, **self.env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.mode(), "today")
        self.assertEqual(self.s.run(self.check, **self.env).returncode, 0)


class InstallFailureSignalTest(unittest.TestCase):
    """CodeBuild skips build and post_build after a failed install command
    but still runs install's finally block, which must send FAILURE."""

    URL = "https://example.com/waitcondition"

    def setUp(self):
        from tests._stubs import AWS_FAKE
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub_py("aws", AWS_FAKE)
        self.s.stub("curl", "exit 0")
        self.s.stub("sha256sum", "cat >/dev/null; exit 1")
        self.spec = load_buildspec("buildspec.yaml")
        self.env = dict(self.spec["env"]["variables"])
        self.env.update(CODEBUILD_SRC_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(INFRA)))),
                        CFN_RESPONSE_URL=self.URL, CODEBUILD_BUILD_ID="team:1")

    def run_install(self, **extra):
        """Run install as CodeBuild does: commands until one fails, then finally."""
        env = dict(self.env)
        env.update(extra)
        install = self.spec["phases"]["install"]
        failed = False
        for cmd in install["commands"]:
            if self.s.run(cmd, **env).returncode != 0:
                failed = True
                break
        for cmd in install["finally"]:
            self.assertEqual(self.s.run(cmd, **env).returncode, 0)
        return failed

    def signals(self):
        return [c for c in self.s.calls("curl") if "-X PUT" in c]

    def test_checksum_mismatch_sends_failure(self):
        self.assertTrue(self.run_install(SLEMIFY_VERSION=""))
        [sig] = self.signals()
        self.assertIn('"Status":"FAILURE"', sig)
        self.assertIn(self.URL, sig)
        self.assertTrue(os.path.exists(os.path.join(self.s.tmp, "cfn-signalled")))
        # post_build, if it ran, would not send a second signal.
        self.s.run(self.spec["phases"]["post_build"]["commands"][1], **self.env)
        self.assertEqual(len(self.signals()), 1)

    def test_missing_slemify_sha_sends_failure(self):
        self.assertTrue(self.run_install(SLEMIFY_VERSION="0.3.0", SLEMIFY_SHA256=""))
        [sig] = self.signals()
        self.assertIn('"Status":"FAILURE"', sig)

    def test_completed_install_sends_nothing(self):
        self.assertEqual(self.spec["phases"]["install"]["commands"][-1], "touch /tmp/install-done")
        open(os.path.join(self.s.tmp, "install-done"), "w").close()
        for cmd in self.spec["phases"]["install"]["finally"]:
            self.assertEqual(self.s.run(cmd, **self.env).returncode, 0)
        self.assertEqual(self.signals(), [])


class PrepFailedMarkerTest(unittest.TestCase):
    def test_failed_json(self):
        from tests._stubs import AWS_FAKE
        s = StubEnv()
        self.addCleanup(s.cleanup)
        s.stub_py("aws", AWS_FAKE)
        with open(os.path.join(s.tmp, "prep-prefix"), "w") as f:
            f.write("releases/v0.3.0/abc/\n")
        with open(os.path.join(s.tmp, "prep-step"), "w") as f:
            f.write("corpus\n")
        spec = load_buildspec("buildspec-prep.yaml")
        # A failed install command would skip post_build; prep.sh does the setup.
        self.assertNotIn("install", spec["phases"])
        cmd = spec["phases"]["post_build"]["commands"][0]
        self.assertIn("retry 5 0 aws s3 cp", cmd)
        r = s.run(cmd, CODEBUILD_BUILD_SUCCEEDING=0, MODEL_BUCKET="central", RELEASE_TAG="v0.3.0",
                  CODEBUILD_SRC_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(INFRA)))),
                  INFRA_DIR="examples/k8s-autoscaling/workshop/infrastructure",
                  CODEBUILD_RESOLVED_SOURCE_VERSION="abc", CODEBUILD_BUILD_ID="p:1",
                  CODEBUILD_BUILD_URL="https://example.com/build")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        failed = json.loads(s.s3_get("central", "releases/v0.3.0/abc/failed.json"))
        self.assertEqual(failed, {"version": 1, "tag": "v0.3.0", "commit": "abc", "step": "corpus",
                                  "buildId": "p:1", "logUrl": "https://example.com/build"})
        # A successful build writes nothing.
        os.remove(s.path("s3", "central", "releases/v0.3.0/abc/failed.json"))
        s.run(cmd, CODEBUILD_BUILD_SUCCEEDING=1, MODEL_BUCKET="central")
        self.assertIsNone(s.s3_get("central", "releases/v0.3.0/abc/failed.json"))


class PinTest(unittest.TestCase):
    PINNED = ("buildspec.yaml", "buildspec-prep.yaml")

    def specs(self):
        for name in self.PINNED:
            if os.path.exists(os.path.join(INFRA, name)):
                yield name, load_buildspec(name)

    def test_sha256_values(self):
        seen = 0
        for name, spec in self.specs():
            for key, value in spec["env"]["variables"].items():
                if key.endswith("_SHA256"):
                    seen += 1
                    self.assertRegex(value, r"^[0-9a-f]{64}$", "%s %s" % (name, key))
        self.assertGreaterEqual(seen, 4)

    def test_downloads_retry_and_no_install_script(self):
        for name, _ in self.specs():
            with open(os.path.join(INFRA, name)) as f:
                text = f.read()
            self.assertNotIn("get-helm-3", text, name)
            for line in text.splitlines():
                if "curl " in line and not line.lstrip().startswith("#"):
                    self.assertIn("--retry", line, "%s: %s" % (name, line.strip()))


class ChecksumGateTest(unittest.TestCase):
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.gate = load_buildspec("buildspec.yaml")["phases"]["install"]["commands"][0]

    def test_gate_is_first_and_fails_without_sha(self):
        self.assertIn("SLEMIFY_SHA256", self.gate)
        r = self.s.run(self.gate, SLEMIFY_VERSION="0.3.0", SLEMIFY_SHA256="")
        self.assertEqual(r.returncode, 1)
        self.assertIn("SLEMIFY_SHA256", r.stdout)

    def test_gate_passes(self):
        self.assertEqual(self.s.run(self.gate, SLEMIFY_VERSION="").returncode, 0)
        self.assertEqual(self.s.run(self.gate, SLEMIFY_VERSION="0.3.0",
                                    SLEMIFY_SHA256="a" * 64).returncode, 0)


if __name__ == "__main__":
    unittest.main()
