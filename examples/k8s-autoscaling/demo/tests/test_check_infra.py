"""scripts/check-infra.sh: central mode, image tags and the GGUF state."""
import os
import shutil
import subprocess
import tempfile
import unittest

from tests import _path

SCRIPT = os.path.join(_path.DEMO_DIR, "scripts", "check-infra.sh")
MIRROR = "222233334444.dkr.ecr.us-west-2.amazonaws.com/slemify"

# A fake aws CLI that answers from environment fixtures:
#   MIRROR          value of /slemify/mirror (unset: ParameterNotFound)
#   GGUF_PRESENT    1 when the GGUF is in the bucket
#   STATUS_<project> status of that project's latest build
#   ECR_MISSING     space-separated repo:tag pairs that are absent
AWS = r'''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as log:
    log.write("aws " + " ".join(args) + "\n")
def opt(name):
    return args[args.index(name) + 1] if name in args else None
svc, op = args[0], args[1]
if svc == "ssm":
    if os.environ.get("MIRROR"):
        print(os.environ["MIRROR"]); sys.exit(0)
    sys.exit(254)
if svc == "eks":
    print("ACTIVE"); sys.exit(0)
if svc == "s3api":
    if os.environ.get("GGUF_PRESENT") == "1":
        print(5000000000); sys.exit(0)
    sys.exit(254)
if svc == "codebuild" and op == "list-builds-for-project":
    print(opt("--project-name") + ":1"); sys.exit(0)
if svc == "codebuild" and op == "batch-get-builds":
    project = opt("--ids").split(":")[0]
    print(os.environ.get("STATUS_" + project.replace("-", "_"), "SUCCEEDED")); sys.exit(0)
if svc == "ecr" and op == "batch-get-image":
    repo = opt("--repository-name")
    i = args.index("--image-ids") + 1
    tags = []
    while i < len(args) and not args[i].startswith("--"):
        tags.append(args[i].split("=", 1)[1]); i += 1
    absent = set(os.environ.get("ECR_MISSING", "").split())
    print("\t".join(t for t in tags if "%s:%s" % (repo, t) in absent)); sys.exit(0)
sys.exit(0)
'''

REPOS = ["slemify/k8s-autoscaling-orchestrator", "slemify/k8s-autoscaling-reranker",
         "slemify/k8s-autoscaling-decision", "slemify/data-pipeline",
         "slemify/classifier-trainer", "slemify/classifier-serving", "slemify/gguf-convert"]


class CheckInfraTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.log = os.path.join(self.dir, "calls.log")
        path = os.path.join(self.dir, "aws")
        with open(path, "w") as f:
            f.write(AWS)
        os.chmod(path, 0o755)

    def run_script(self, **env):
        open(self.log, "w").close()
        e = {"PATH": self.dir + ":" + os.environ["PATH"], "STUB_LOG": self.log,
             "SLEMIFY_BUCKET": "team-models", "AWS_REGION": "us-west-2", "GGUF_PRESENT": "1"}
        e.update(env)
        r = subprocess.run(["bash", SCRIPT], env=e, capture_output=True, text=True, timeout=60)
        with open(self.log) as f:
            calls = f.read().splitlines()
        return r, calls

    def ecr_calls(self, calls):
        return [c for c in calls if c.startswith("aws ecr batch-get-image")]

    def test_central_mode_checks_latest_and_llama(self):
        r, calls = self.run_script(MIRROR=MIRROR, SLEMIFY_MIRROR_REGISTRY=MIRROR)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        ecr = self.ecr_calls(calls)
        self.assertEqual(len(ecr), 8)
        for c in ecr[:7]:
            self.assertIn("--image-ids imageTag=latest --query", c)
        self.assertIn("--repository-name slemify/llama-cpp --image-ids imageTag=server-b12063", ecr[7])
        self.assertIn("[ok]      Images: all 8 tags in ECR", r.stdout)
        self.assertIn("All ready.", r.stdout)
        self.assertNotIn("[fix]", r.stdout)

    def test_today_checks_three_tags(self):
        r, calls = self.run_script(ECR_MISSING="slemify/data-pipeline:latest-arm64")
        self.assertEqual(r.returncode, 1)
        ecr = self.ecr_calls(calls)
        self.assertEqual(len(ecr), 7)
        for c in ecr:
            self.assertIn("imageTag=latest imageTag=latest-amd64 imageTag=latest-arm64", c)
        self.assertIn("  [missing] Image slemify/data-pipeline:latest-arm64", r.stdout)
        self.assertNotIn("llama-cpp", r.stdout)
        self.assertEqual(r.stdout.count("[missing]"), 1)

    def test_central_gguf_wait(self):
        r, _ = self.run_script(MIRROR=MIRROR, SLEMIFY_MIRROR_REGISTRY=MIRROR, GGUF_PRESENT="0",
                               STATUS_slemify_workshop_provision="IN_PROGRESS")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("  [wait]    Analyst GGUF: provisioning is still copying it. About 10 minutes; run this again.",
                      r.stdout)

    def test_central_gguf_missing(self):
        r, _ = self.run_script(MIRROR=MIRROR, SLEMIFY_MIRROR_REGISTRY=MIRROR, GGUF_PRESENT="0",
                               STATUS_slemify_workshop_provision="FAILED")
        self.assertEqual(r.returncode, 1)
        self.assertIn("  [missing] Analyst GGUF: not in the bucket, and the last provisioning build ended FAILED",
                      r.stdout)

    def test_fix_line_when_terminal_predates_mirror(self):
        r, _ = self.run_script(MIRROR=MIRROR)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("  [fix]     This terminal started before provisioning copied llama.cpp. Run: source ~/.bashrc",
                      r.stdout)

    def test_today_gguf_messages_unchanged(self):
        r, _ = self.run_script(GGUF_PRESENT="0", STATUS_slemify_workshop_gguf="IN_PROGRESS")
        self.assertIn("conversion still running in CodeBuild", r.stdout)


if __name__ == "__main__":
    unittest.main()
