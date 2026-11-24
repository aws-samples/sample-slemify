# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Run buildspec command blocks with stub docker, aws, crane and sleep."""
import json
import os
import re
import unittest

import yaml

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


# A terraform stub for the targeted applies. "apply" fails while the number of
# earlier applies is below $TF_FAILS ("always" fails every time). With
# $TF_SLOW set, an apply that targets module.eks runs for a second and writes
# "eks-finished" only when it is left to run to the end.
TF_RETRY_STUB = r'''
case "$1" in
  output) echo team-models; exit 0 ;;
  apply)
    n=$(cat "$STUB_DIR/tf-applies" 2>/dev/null || echo 0)
    echo $((n + 1)) > "$STUB_DIR/tf-applies"
    if [ "${TF_SLOW:-0}" = 1 ]; then
      /bin/sleep 1
      touch "$STUB_DIR/eks-finished"
    fi
    if [ "${TF_FAILS:-0}" = always ] || [ "$n" -lt "${TF_FAILS:-0}" ]; then
      echo "Error: reading ec2 resource: empty result" >&2
      exit 1
    fi
    exit 0 ;;
esac
exit 0
'''
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(INFRA))))
INFRA_REL = "examples/k8s-autoscaling/workshop/infrastructure"
class TargetedApplyRetryTest(unittest.TestCase):
    """The VPC, model bucket and EKS applies retry like the full apply."""
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub("terraform", TF_RETRY_STUB)
        self.spec = load_buildspec("buildspec.yaml")
        self.env = dict(CODEBUILD_SRC_DIR=REPO_ROOT, INFRA_DIR=INFRA_REL)
    def applies(self):
        return self.s.calls("terraform apply")
    def test_the_three_targeted_applies_use_retry_3(self):
        for needle in ("-target=module.vpc", "-target=aws_s3_bucket.models", "-target=module.eks"):
            cmd = find_command(self.spec, "build", needle)
            self.assertIn("retry 3 0 bash", cmd, needle)
            self.assertIn("lib/tf-apply.sh", cmd, needle)
            self.assertNotIn("\nterraform apply", "\n" + cmd, needle)
    def test_vpc_apply_recovers_from_one_failure(self):
        cmd = find_command(self.spec, "build", "-target=module.vpc")
        r = self.s.run(cmd, TF_FAILS=1, **self.env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.applies()), 2)
        self.assertTrue(all("-target=module.vpc" in c for c in self.applies()))
        self.assertIn("attempt 1 of 3", r.stderr)
        self.assertEqual(len(self.s.calls("sleep")), 1)
    def test_models_apply_fails_after_three_attempts(self):
        cmd = find_command(self.spec, "build", "-target=aws_s3_bucket.models")
        r = self.s.run(cmd, TF_FAILS="always", **self.env)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(self.applies()), 3)
        self.assertEqual(len(self.s.calls("sleep")), 2)
        self.assertIn("-target=aws_s3_bucket_versioning.models", self.applies()[0])
    def test_eks_apply_in_the_background_recovers(self):
        cmd = find_command(self.spec, "build", "-target=module.eks")
        r = self.s.run(cmd, TF_FAILS=1, **self.env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.applies()), 2)
        with open(os.path.join(self.s.tmp, "eks-apply.log")) as f:
            log = f.read()
        self.assertIn("retry: bash", log)
        self.assertNotIn("EKS apply failed", r.stdout)
    def test_eks_apply_failure_reaches_wait_and_fails_the_build(self):
        cmd = find_command(self.spec, "build", "-target=module.eks")
        r = self.s.run(cmd, TF_FAILS="always", **self.env)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(self.applies()), 3)
        self.assertIn("EKS apply failed", r.stdout)
        self.assertIn("empty result", r.stdout)
    def test_failed_amd64_build_waits_for_the_eks_apply(self):
        # The block used to kill the background apply, which left terraform
        # running under a dead subshell and the state half written.
        self.s.stub("aws", 'case "$1 $2" in "codebuild batch-get-builds") echo FAILED ;; esac\nexit 0')
        with open(os.path.join(self.s.tmp, "image-builds"), "w") as f:
            f.write("amd:1 arm:1\n")
        cmd = find_command(self.spec, "build", "-target=module.eks")
        r = self.s.run(cmd, TF_SLOW=1, GGUF_BUILD="gguf", **self.env)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("amd64 image build failed", r.stdout)
        self.assertIn("waiting for the EKS apply to finish before failing", r.stdout)
        self.assertTrue(os.path.exists(self.s.path("eks-finished")))
        self.assertNotIn("kill", cmd)
class PreBuildSignalTest(unittest.TestCase):
    """A failed pre_build command skips the build phase and post_build's
    signal, so pre_build's finally block sends FAILURE."""
    URL = "https://example.com/waitcondition"
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub("curl", "exit 0")
        self.spec = load_buildspec("buildspec.yaml")
        self.pre = self.spec["phases"]["pre_build"]
        self.env = dict(CFN_RESPONSE_URL=self.URL, CODEBUILD_BUILD_ID="team:1")
    def run_finally(self, **extra):
        env = dict(self.env)
        env.update(extra)
        for cmd in self.pre["finally"]:
            self.assertEqual(self.s.run(cmd, **env).returncode, 0)
    def signals(self):
        return [c for c in self.s.calls("curl") if "-X PUT" in c]
    def test_the_last_pre_build_command_marks_it_done(self):
        self.assertEqual(self.pre["commands"][-1], "touch /tmp/prebuild-done")
    def test_unfinished_pre_build_sends_failure(self):
        self.run_finally()
        [sig] = self.signals()
        self.assertIn('"Status":"FAILURE"', sig)
        self.assertIn("pre_build", sig)
        self.assertIn(self.URL, sig)
        self.assertTrue(os.path.exists(os.path.join(self.s.tmp, "cfn-signalled")))
        # post_build, if it ran, would not send a second signal.
        self.s.run(self.spec["phases"]["post_build"]["commands"][1], **self.env)
        self.assertEqual(len(self.signals()), 1)
    def test_finished_pre_build_sends_nothing(self):
        open(os.path.join(self.s.tmp, "prebuild-done"), "w").close()
        self.run_finally()
        self.assertEqual(self.signals(), [])
    def test_no_second_signal_after_an_install_failure(self):
        open(os.path.join(self.s.tmp, "cfn-signalled"), "w").close()
        self.run_finally()
        self.assertEqual(self.signals(), [])
    def test_no_signal_without_a_response_url(self):
        self.run_finally(CFN_RESPONSE_URL="")
        self.assertEqual(self.signals(), [])
    def test_a_failing_terraform_init_ends_pre_build_without_the_marker(self):
        # Run the real pre_build commands as CodeBuild does, with a failing init.
        self.s.stub("terraform", 'exit 1')
        self.s.stub("aws", "echo build:1")
        with open(os.path.join(self.s.tmp, "central-mode"), "w") as f:
            f.write("central\n")
        os.makedirs(os.path.join(self.s.root, "src", INFRA_REL, "terraform"))
        env = dict(self.spec["env"]["variables"])
        env.update(self.env, CODEBUILD_SRC_DIR=os.path.join(self.s.root, "src"), TF_STATE_BUCKET="b",
                   STACK_NAME="workshop-stack", EKS_CLUSTER_NAME="c", EKS_CLUSTER_VERSION="1.36")
        # The first failed command stops the phase.
        r = self.s.run("set -e\n" + "\n".join(self.pre["commands"]) + "\n", **env)
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.s.calls("terraform"), ["terraform init -input=false"])
        self.assertFalse(os.path.exists(os.path.join(self.s.tmp, "prebuild-done")))
        self.run_finally()
        self.assertEqual(len(self.signals()), 1)
class FaultHookTest(unittest.TestCase):
    """provision-fail-once fails the build before the seed, once per account."""
    def setUp(self):
        from tests._stubs import AWS_FAKE
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub_py("aws", AWS_FAKE)
        spec = load_buildspec("buildspec.yaml")
        self.cmd = find_command(spec, "build", "provision-fail-once")
        self.env = dict(CODEBUILD_SRC_DIR=REPO_ROOT, INFRA_DIR=INFRA_REL)
    def test_off_by_default(self):
        r = self.s.run(self.cmd, **self.env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.s.calls("aws"), [])
    def test_fails_once_and_records_it(self):
        r = self.s.run(self.cmd, FAULT_INJECTION="provision-fail-once", **self.env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAULT INJECTION: provision-fail-once", r.stdout)
        self.assertEqual(len(self.s.calls("aws ssm put-parameter --name /slemify/fault/provision-fail-once")), 1)
    def test_does_not_fire_again_once_recorded(self):
        self.s.state("param", "fired")
        r = self.s.run(self.cmd, FAULT_INJECTION="provision-fail-once", **self.env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.s.calls("aws ssm put-parameter"), [])
    def test_runs_after_the_full_apply_and_before_the_seed(self):
        spec = load_buildspec("buildspec.yaml")
        cmds = phase_commands(spec, "build")
        idx = [i for i, c in enumerate(cmds) if "provision-fail-once" in c]
        self.assertEqual(len(idx), 1)
        full = [i for i, c in enumerate(cmds) if "enable_s3_csi_addon=false" in c][0]
        seed = [i for i, c in enumerate(cmds) if c.strip() == "bash seed/seed.sh"][0]
        self.assertLess(full, idx[0])
        self.assertLess(idx[0], seed)
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
    PINNED = ("buildspec.yaml", "buildspec-prep.yaml", "buildspec-destroy.yaml")
    # Shell files outside the buildspecs that download tools.
    SCRIPTS = ("lib/prep.sh",)

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

    def commands(self):
        """Each file's non-comment lines, with backslash continuations joined
        so one command is one line."""
        for name in self.PINNED + self.SCRIPTS:
            with open(os.path.join(INFRA, name)) as f:
                text = f.read().replace("\\\n", " ")
            yield name, [l for l in text.splitlines() if not l.lstrip().startswith("#")]

    def test_downloads_retry_and_no_install_script(self):
        for name, lines in self.commands():
            self.assertNotIn("get-helm-3", "\n".join(lines), name)
            for line in lines:
                if "curl " in line:
                    self.assertIn("--retry", line, "%s: %s" % (name, line.strip()))

    def test_downloads_are_checked(self):
        # A download writes to a file with -o, and a later sha256sum -c line
        # names that file. The stack signals (curl -X PUT) are not downloads.
        downloads = 0
        for name, lines in self.commands():
            for line in lines:
                if "curl " not in line or "-X PUT" in line:
                    continue
                m = re.search(r"\s-o\s+(\S+)", line)
                self.assertIsNotNone(m, "%s: download without -o: %s" % (name, line.strip()))
                path = m.group(1).strip("\"'")
                downloads += 1
                self.assertTrue(any("sha256sum -c" in l and path in l for l in lines),
                                "%s: no sha256sum -c for %s" % (name, path))
        # buildspec.yaml 5, buildspec-destroy.yaml 2, lib/prep.sh 1.
        self.assertGreaterEqual(downloads, 8)

    def test_destroy_pins_match_the_build(self):
        build = load_buildspec("buildspec.yaml")["env"]["variables"]
        destroy = load_buildspec("buildspec-destroy.yaml")["env"]["variables"]
        for key in ("TERRAFORM_VERSION", "TERRAFORM_SHA256", "KUBECTL_VERSION", "KUBECTL_SHA256"):
            self.assertEqual(destroy[key], build[key], key)


# An aws stub for the destroy build. Every role exists, carries one attached
# policy and one inline policy, and every other call succeeds.
AWS_IAM_PRESENT = r'''
case "$1 $2" in
  "iam list-attached-role-policies") echo "arn:aws:iam::aws:policy/ReadOnlyAccess" ;;
  "iam list-role-policies") echo "slemify-data-pipeline" ;;
esac
exit 0
'''
# The same, with no role in the account: a run that stopped before module 1.
AWS_IAM_MISSING = r'''
case "$1 $2" in
  "iam get-role")
    echo "An error occurred (NoSuchEntity) when calling the GetRole operation" >&2
    exit 254 ;;
esac
exit 0
'''
# get-role fails for a reason other than a missing role: an expired session.
# The role may well be there, so this must not be read as "nothing to delete".
AWS_IAM_EXPIRED = r'''
case "$1 $2" in
  "iam get-role")
    echo "An error occurred (ExpiredTokenException) when calling the GetRole operation" >&2
    exit 254 ;;
esac
exit 0
'''
# The role is there and its policies come off, but the role itself does not.
AWS_IAM_DELETE_DENIED = r'''
case "$1 $2" in
  "iam list-attached-role-policies") echo "arn:aws:iam::aws:policy/ReadOnlyAccess" ;;
  "iam list-role-policies") echo "slemify-data-pipeline" ;;
  "iam delete-role")
    echo "An error occurred (AccessDenied) when calling the DeleteRole operation" >&2
    exit 254 ;;
esac
exit 0
'''


class DestroyIamTest(unittest.TestCase):
    """buildspec-destroy.yaml deletes the data-pipeline roles. Terraform never
    created them, so terraform destroy leaves them in the account."""

    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(INFRA))))
    INFRA_DIR = "examples/k8s-autoscaling/workshop/infrastructure"

    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        spec = load_buildspec("buildspec-destroy.yaml")
        self.roles = spec["env"]["variables"]["DATA_ROLES"].split()
        self.cmd = find_command(spec, "build", "terraform destroy")
        self.s.stub("terraform", 'case "$1" in output) echo team-models ;; esac\nexit 0')
        self.s.stub("kubectl", "exit 0")
        # Emptying the versioned buckets needs boto3 and real credentials.
        self.s.stub("python3", "exit 0")

    def run_block(self, aws_body):
        self.s.stub("aws", aws_body)
        return self.s.run(self.cmd, CODEBUILD_SRC_DIR=self.REPO, INFRA_DIR=self.INFRA_DIR,
                          NODEPOOLS="slemify-slm", DATA_ROLES=" ".join(self.roles),
                          EKS_CLUSTER_NAME="workshop-cluster", TF_STATE_BUCKET="state-bucket",
                          STACK_NAME="workshop-stack", AWS_REGION="us-west-2")

    def test_role_names_match_the_example_configs(self):
        examples = os.path.dirname(os.path.dirname(INFRA))
        expected = []
        for project in ("analyst", "retriever", "triage"):
            with open(os.path.join(examples, project, "expert.yaml")) as f:
                expected.append("slemify-%s-data" % yaml.safe_load(f)["project"]["name"])
        self.assertEqual(sorted(self.roles), sorted(expected))

    def test_each_role_is_detached_and_deleted(self):
        r = self.run_block(AWS_IAM_PRESENT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("destroy complete", r.stdout)
        calls = self.s.calls("aws iam")
        for role in self.roles:
            self.assertIn("aws iam detach-role-policy --role-name %s "
                          "--policy-arn arn:aws:iam::aws:policy/ReadOnlyAccess" % role, calls)
            self.assertIn("aws iam delete-role-policy --role-name %s "
                          "--policy-name slemify-data-pipeline" % role, calls)
            self.assertIn("aws iam delete-role --role-name %s" % role, calls)

    def test_a_missing_role_does_not_fail_the_build(self):
        r = self.run_block(AWS_IAM_MISSING)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("destroy complete", r.stdout)
        for role in self.roles:
            self.assertIn("role %s not found" % role, r.stdout)
        self.assertEqual([c for c in self.s.calls("aws iam") if "delete-role" in c], [])

    def test_an_iam_error_that_is_not_a_missing_role_fails_the_build(self):
        r = self.run_block(AWS_IAM_EXPIRED)
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("destroy complete", r.stdout)
        self.assertNotIn("not found", r.stdout)
        self.assertIn("could not read role", r.stdout + r.stderr)

    def test_a_role_that_cannot_be_deleted_fails_the_build(self):
        r = self.run_block(AWS_IAM_DELETE_DENIED)
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("destroy complete", r.stdout)
        self.assertNotIn("deleted role", r.stdout)

    def test_the_roles_go_after_terraform_destroy(self):
        # The cluster's Pod Identity associations reference these roles.
        block = self.cmd
        self.assertLess(block.index("terraform destroy"), block.index("${DATA_ROLES}"))


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
