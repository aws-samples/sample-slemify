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

    def run(self, script, timeout=60, **extra):
        """Run a bash script string; /tmp/ in it is redirected to the temp dir."""
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
