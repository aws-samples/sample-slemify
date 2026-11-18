# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Helpers that run buildspec blocks and shell libraries with stub commands.

Each stub appends one line, "<name> <args>", to $STUB_LOG and then runs the
body the test gives it. Stubs sit first on PATH, so the real tools are never
called.
"""
import os
import shutil
import subprocess
import tempfile

import yaml

INFRA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_buildspec(name):
    with open(os.path.join(INFRA, name)) as f:
        return yaml.safe_load(f)


def phase_commands(spec, phase):
    return spec["phases"][phase]["commands"]


def find_command(spec, phase, needle):
    for cmd in phase_commands(spec, phase):
        if needle in cmd:
            return cmd
    raise AssertionError("no %s command contains %r" % (phase, needle))


COMMIT = "0123456789abcdef0123456789abcdef01234567"
TAG = "v0.3.0"
PREFIX = "releases/%s/%s/" % (TAG, COMMIT)
LLAMA_DIGEST = "sha256:10a38e6cc91760ab157f0a0a8546218d35f4a314432a70298abfa3d096829ad7"
REPOS = ["slemify/k8s-autoscaling-orchestrator", "slemify/k8s-autoscaling-reranker",
         "slemify/k8s-autoscaling-decision", "slemify/data-pipeline",
         "slemify/classifier-trainer", "slemify/classifier-serving",
         "slemify/gguf-convert", "slemify/llama-cpp"]

# A fake aws CLI. S3 objects are files under $STUB_DIR/s3/<bucket>/<key>.
# Behavior comes from files under $STUB_DIR/state:
#   param               value of the SSM parameter (absent: ParameterNotFound)
#   param-error         stderr for a failed get-parameter that is not "not found"
#   head-bucket-error   stderr for a failed head-bucket
#   have-tags           ecr describe-images succeeds
#   gguf-builds         ids list-builds-for-project prints
#   inprogress-<id>     how many more times batch-get-builds reports IN_PROGRESS
#   status-<id>         final status (default SUCCEEDED)
#   s3-copy-fail        every S3-to-S3 copy fails
# A start-build that passes KEY_PREFIX (the GGUF build) writes the GGUF object.
AWS_FAKE = r'''#!/usr/bin/env python3
import os, shutil, sys
d = os.environ["STUB_DIR"]
st = os.path.join(d, "state")
s3 = os.path.join(d, "s3")
args = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as log:
    log.write("aws " + " ".join(args) + "\n")
def read(name, default=None):
    try:
        with open(os.path.join(st, name)) as f:
            return f.read()
    except OSError:
        return default
def opt(name):
    return args[args.index(name) + 1] if name in args else None
def s3path(uri):
    return os.path.join(s3, uri[len("s3://"):])
def fail(msg, code=254):
    sys.stderr.write(msg + "\n")
    sys.exit(code)
svc, op = args[0], args[1]
if svc == "ssm" and op == "get-parameter":
    v = read("param")
    if v is not None:
        print(v)
        sys.exit(0)
    err = read("param-error")
    if err:
        fail(err, 255)
    fail("An error occurred (ParameterNotFound) when calling the GetParameter operation")
if svc == "ssm":
    sys.exit(0)
if svc == "sts":
    print("222233334444")
    sys.exit(0)
if svc == "ecr" and op == "get-login-password":
    print("token")
    sys.exit(0)
if svc == "ecr" and op == "describe-images":
    sys.exit(0 if read("have-tags") is not None else 1)
if svc == "ecr":
    sys.exit(0)
if svc == "s3api" and op == "head-bucket":
    err = read("head-bucket-error")
    if err:
        fail(err)
    sys.exit(0)
if svc == "s3api" and op == "head-object":
    path = os.path.join(s3, opt("--bucket"), opt("--key"))
    if not os.path.exists(path):
        fail("An error occurred (404) when calling the HeadObject operation: Not Found")
    if opt("--query") == "ContentLength":
        print(os.path.getsize(path))
    sys.exit(0)
if svc == "s3" and op in ("cp", "rm"):
    pos, skip = [], False
    for a in args[2:]:
        if skip:
            skip = False
        elif a == "--content-type":
            skip = True
        elif not a.startswith("--"):
            pos.append(a)
    if op == "rm":
        if os.path.exists(s3path(pos[0])):
            os.remove(s3path(pos[0]))
        sys.exit(0)
    src, dst = pos[0], pos[1]
    if src.startswith("s3://") and dst.startswith("s3://") and read("s3-copy-fail") is not None:
        fail("copy failed", 1)
    srcp = s3path(src) if src.startswith("s3://") else src
    if not os.path.exists(srcp):
        fail("An error occurred (404) when calling the HeadObject operation: Key not found", 1)
    dstp = s3path(dst) if dst.startswith("s3://") else dst
    os.makedirs(os.path.dirname(dstp) or ".", exist_ok=True)
    shutil.copyfile(srcp, dstp)
    sys.exit(0)
if svc == "codebuild" and op == "start-build":
    project = opt("--project-name")
    over = {}
    for a in args:
        if a.startswith("name=") and ",value=" in a:
            name, rest = a[5:].split(",value=", 1)
            over[name] = rest.rsplit(",type=", 1)[0]
    if "KEY_PREFIX" in over:
        path = os.path.join(s3, over["MODEL_BUCKET"], over["KEY_PREFIX"] + "models/k8s-autoscaling-analyst/model-q4_k_m.gguf")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("GGUF")
    print(project + ":1")
    sys.exit(0)
if svc == "codebuild" and op == "list-builds-for-project":
    print(read("gguf-builds", "None").strip())
    sys.exit(0)
if svc == "codebuild" and op == "batch-get-builds":
    i = args.index("--ids") + 1
    ids = []
    while i < len(args) and not args[i].startswith("--"):
        ids.append(args[i])
        i += 1
    out = []
    for bid in ids:
        key = bid.replace(":", "_")
        n = int(read("inprogress-" + key, "0"))
        if n > 0:
            with open(os.path.join(st, "inprogress-" + key), "w") as f:
                f.write(str(n - 1))
            out.append("IN_PROGRESS")
        else:
            out.append((read("status-" + key) or "SUCCEEDED").strip())
    print("\t".join(out))
    sys.exit(0)
sys.exit(0)
'''

# crane: "copy" fails when the source contains $CRANE_FAIL; "digest" prints
# the pinned digest for llama-cpp and a fixed one otherwise.
CRANE_STUB = r'''
case "$1" in
  auth) cat >/dev/null; exit 0 ;;
  copy) if [ -n "${CRANE_FAIL:-}" ] && [ "${2#*"$CRANE_FAIL"}" != "$2" ]; then exit 1; fi; exit 0 ;;
  digest) case "$2" in
      *llama-cpp*) echo "sha256:10a38e6cc91760ab157f0a0a8546218d35f4a314432a70298abfa3d096829ad7" ;;
      *) echo "sha256:$(printf 'b%.0s' $(seq 1 64))" ;;
    esac ;;
esac
exit 0
'''

TERRAFORM_STUB = r'''
case "$1" in
  output) echo team-models ;;
  apply) [ "${TF_APPLY_FAIL:-0}" = 1 ] && exit 1; exit 0 ;;
esac
exit 0
'''


class StubEnv:
    """A temp dir with a stub bin dir, a call log and a clean environment."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="stubenv-")
        self.bin = os.path.join(self.root, "bin")
        self.tmp = os.path.join(self.root, "tmp")
        os.makedirs(self.bin)
        os.makedirs(self.tmp)
        self.log = os.path.join(self.root, "calls.log")
        open(self.log, "w").close()
        # Time-wasting commands are no-ops by default.
        self.stub("sleep", "exit 0")
        # timeout <seconds> <cmd...>: run the command without a limit.
        self.stub("timeout", 'shift; exec "$@"', log=False)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def stub(self, name, body, log=True):
        path = os.path.join(self.bin, name)
        lines = ["#!/bin/bash"]
        if log:
            lines.append('echo "%s $*" >> "$STUB_LOG"' % name)
        lines.append(body)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(path, 0o755)

    def stub_py(self, name, source):
        path = os.path.join(self.bin, name)
        with open(path, "w") as f:
            f.write(source)
        os.chmod(path, 0o755)

    def state(self, name, value):
        """Write a state file the fake aws reads (see AWS_FAKE)."""
        os.makedirs(self.path("state"), exist_ok=True)
        with open(self.path("state", name), "w") as f:
            f.write(value)

    def s3_put(self, bucket, key, data):
        path = self.path("s3", bucket, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(data)
        return path

    def s3_get(self, bucket, key):
        path = self.path("s3", bucket, key)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return f.read()

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def env(self, **extra):
        env = {
            "PATH": self.bin + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": self.root,
            "STUB_LOG": self.log,
            "STUB_DIR": self.root,
            "CENTRAL_TMP": self.tmp,
            "AWS_REGION": "us-west-2",
            "LANG": "C",
        }
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def run(self, script, timeout=60, redirect_tmp=True, **extra):
        """Run a bash script string. With redirect_tmp, /tmp/ in a buildspec
        block is redirected to the temp dir."""
        if redirect_tmp:
            script = script.replace("/tmp/", self.tmp + "/")
        return subprocess.run(
            ["bash", "-c", script],
            env=self.env(**extra),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def calls(self, prefix=""):
        with open(self.log) as f:
            return [line.rstrip("\n") for line in f if line.startswith(prefix)]
