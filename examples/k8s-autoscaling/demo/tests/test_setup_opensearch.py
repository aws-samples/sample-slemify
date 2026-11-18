"""scripts/setup-opensearch.sh: helm repo update is retried 3 times."""
import os
import shutil
import subprocess
import tempfile
import unittest

from tests import _path

SCRIPT = os.path.join(_path.DEMO_DIR, "scripts", "setup-opensearch.sh")

# helm stub: "repo update" fails until it has been called $UPDATE_FAILS times.
HELM = r'''#!/bin/bash
echo "helm $*" >> "$STUB_LOG"
if [ "$1 $2" = "repo update" ]; then
  n=$(cat "$STUB_DIR/updates" 2>/dev/null || echo 0)
  n=$((n + 1)); echo "$n" > "$STUB_DIR/updates"
  [ "$n" -le "${UPDATE_FAILS:-0}" ] && exit 1
fi
exit 0
'''


class SetupOpenSearchTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.log = os.path.join(self.dir, "calls.log")
        open(self.log, "w").close()
        stubs = {
            "helm": HELM,
            "kubectl": '#!/bin/bash\necho "kubectl $*" >> "$STUB_LOG"\nexit 0\n',
            "sleep": '#!/bin/bash\necho "sleep $*" >> "$STUB_LOG"\nexit 0\n',
        }
        for name, body in stubs.items():
            path = os.path.join(self.dir, name)
            with open(path, "w") as f:
                f.write(body)
            os.chmod(path, 0o755)

    def run_script(self, update_fails):
        env = {"PATH": self.dir + ":" + os.environ["PATH"], "STUB_LOG": self.log,
               "STUB_DIR": self.dir, "UPDATE_FAILS": str(update_fails)}
        r = subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60)
        with open(self.log) as f:
            calls = f.read().splitlines()
        return r, calls

    def test_two_failures_then_success(self):
        r, calls = self.run_script(2)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len([c for c in calls if c.startswith("helm repo update")]), 3)
        self.assertEqual(len([c for c in calls if c == "sleep 30"]), 2)
        self.assertTrue(any(c.startswith("helm upgrade --install") for c in calls))

    def test_always_failing_stops(self):
        r, calls = self.run_script(99)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(len([c for c in calls if c.startswith("helm repo update")]), 3)
        self.assertFalse(any(c.startswith("helm upgrade") for c in calls))


if __name__ == "__main__":
    unittest.main()
