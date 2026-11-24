# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""lib/fault.sh and lib/tf-apply.sh with a stub aws and terraform."""
import hashlib
import json
import os
import time
import unittest
from tests._stubs import AWS_FAKE, INFRA, StubEnv
FAULT = os.path.join(INFRA, "lib", "fault.sh")
TF_APPLY = os.path.join(INFRA, "lib", "tf-apply.sh")
ACCOUNT = "222233334444"  # what the fake aws prints for sts get-caller-identity
def fraction(account):
    return int(hashlib.sha256(account.encode()).hexdigest()[:8], 16) / float(16 ** 8)
class FaultShTest(unittest.TestCase):
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub_py("aws", AWS_FAKE)
    def sh(self, body, **env):
        return self.s.run('. "%s"\n%s' % (FAULT, body), redirect_tmp=False, **env)
    def on(self, name, once="", **env):
        return self.sh("fault_on %s %s" % (name, once), **env).returncode == 0
    def test_off_when_unset_or_empty(self):
        self.assertFalse(self.on("tf-eks-fail-first"))
        self.assertFalse(self.on("tf-eks-fail-first", FAULT_INJECTION=""))
        self.assertEqual(self.s.calls("aws"), [])
    def test_listed_names_are_on(self):
        env = {"FAULT_INJECTION": "tf-eks-fail-first, no-central-pairs,bedrock-unavailable:600"}
        self.assertTrue(self.on("tf-eks-fail-first", **env))
        self.assertTrue(self.on("no-central-pairs", **env))
        self.assertTrue(self.on("bedrock-unavailable", **env))
        self.assertFalse(self.on("provision-fail-once", **env))
        # A name that only starts like a listed one is not on.
        self.assertFalse(self.on("tf-eks", **env))
        self.assertEqual(self.s.calls("aws"), [])
    def test_argument(self):
        r = self.sh("fault_arg bedrock-unavailable", FAULT_INJECTION="no-central-pairs,bedrock-unavailable:600@1")
        self.assertEqual(r.stdout.strip(), "600")
    def test_share_picks_accounts_by_hash(self):
        f = fraction(ACCOUNT)
        above = "%.6f" % min(1.0, f + 0.001)
        below = "%.6f" % max(0.0, f - 0.001)
        self.assertTrue(self.on("no-central-pairs", FAULT_INJECTION="no-central-pairs@" + above))
        self.assertFalse(self.on("no-central-pairs", FAULT_INJECTION="no-central-pairs@" + below))
        self.assertTrue(any("sts get-caller-identity" in c for c in self.s.calls("aws")))
    def test_share_zero_and_one_need_no_account_lookup(self):
        self.assertFalse(self.on("no-central-pairs", FAULT_INJECTION="no-central-pairs@0"))
        self.assertTrue(self.on("no-central-pairs", FAULT_INJECTION="no-central-pairs@1"))
        self.assertEqual(self.s.calls("aws"), [])
    def test_bad_share_turns_the_fault_off(self):
        r = self.sh("fault_on no-central-pairs", FAULT_INJECTION="no-central-pairs@2")
        self.assertEqual(r.returncode, 1)
        self.assertIn("not a number from 0 to 1", r.stderr)
        self.assertFalse(self.on("no-central-pairs", FAULT_INJECTION="no-central-pairs@abc"))
    def test_once_fires_until_it_is_recorded(self):
        env = {"FAULT_INJECTION": "provision-fail-once"}
        self.assertTrue(self.on("provision-fail-once", "once", **env))
        r = self.sh("fault_fired provision-fail-once", **env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(any("ssm put-parameter --name /slemify/fault/provision-fail-once" in c
                            for c in self.s.calls("aws")))
        self.s.state("param", "fired")  # the fake aws now finds the parameter
        self.assertFalse(self.on("provision-fail-once", "once", **env))
    def test_once_does_not_fire_when_the_parameter_cannot_be_read(self):
        self.s.state("param-error", "An error occurred (ThrottlingException) when calling the GetParameter operation")
        self.assertFalse(self.on("provision-fail-once", "once", FAULT_INJECTION="provision-fail-once"))
    def test_bedrock_window_is_written_and_removed(self):
        env = {"FAULT_INJECTION": "bedrock-unavailable:600"}
        before = int(time.time())
        r = self.sh('fault_bedrock_start team-models\nfault_bedrock_stop team-models', **env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("FAULT INJECTION", r.stdout)
        puts = [c for c in self.s.calls("aws s3 cp") if "_fault/bedrock.json" in c]
        self.assertEqual(len(puts), 1)
        self.assertIn("s3://team-models/k8s-autoscaling/data/_fault/bedrock.json", puts[0])
        self.assertEqual(len(self.s.calls("aws s3 rm")), 1)
        # The window file itself, as it was uploaded.
        with open(os.path.join(self.s.tmp, "fault-bedrock.json")) as f:
            until = json.load(f)["until"]
        self.assertTrue(before + 600 <= until <= int(time.time()) + 600)
    def test_bedrock_window_needs_the_fault_and_a_number(self):
        self.assertEqual(self.sh("fault_bedrock_start team-models").returncode, 0)
        r = self.sh("fault_bedrock_start team-models", FAULT_INJECTION="bedrock-unavailable:soon")
        self.assertEqual(r.returncode, 0)
        self.assertIn("needs a number of seconds", r.stderr)
        self.assertEqual(self.s.calls("aws s3"), [])
        # Stop does nothing when nothing was written.
        self.sh("fault_bedrock_stop team-models", FAULT_INJECTION="bedrock-unavailable:600")
        self.assertEqual(self.s.calls("aws s3 rm"), [])
class TfApplyTest(unittest.TestCase):
    def setUp(self):
        self.s = StubEnv()
        self.addCleanup(self.s.cleanup)
        self.s.stub_py("aws", AWS_FAKE)
        self.s.stub("terraform", "exit 0")
    def apply(self, *args, **env):
        return self.s.run('bash "%s" %s' % (TF_APPLY, " ".join(args)), redirect_tmp=False, **env)
    def test_runs_terraform_apply_with_the_arguments(self):
        r = self.apply("-target=module.vpc")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.s.calls("terraform"),
                         ["terraform apply -input=false -auto-approve -target=module.vpc"])
    def test_exit_status_comes_from_terraform(self):
        self.s.stub("terraform", "exit 3")
        self.assertEqual(self.apply("-var enable_s3_csi_addon=false").returncode, 3)
    def test_eks_fault_fails_only_the_first_eks_attempt(self):
        env = {"FAULT_INJECTION": "tf-eks-fail-first"}
        r = self.apply("-target=module.vpc", **env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.s.calls("terraform")), 1)
        r = self.apply("-target=module.eks", **env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAULT INJECTION: tf-eks-fail-first", r.stderr)
        self.assertEqual(len(self.s.calls("terraform")), 1)
        r = self.apply("-target=module.eks", **env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.s.calls("terraform")), 2)
    def test_no_fault_without_the_setting(self):
        for _ in range(2):
            self.assertEqual(self.apply("-target=module.eks").returncode, 0)
        self.assertEqual(len(self.s.calls("terraform")), 2)
if __name__ == "__main__":
    unittest.main()
