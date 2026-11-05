"""scripts/index-knowledge.py: _with_retry outlasts a dropped port-forward and
does not retry a timed-out insert."""
import io
import sys
import types
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest import mock

import httpx
from opensearchpy.exceptions import ConnectionError as OSConnectionError, ConnectionTimeout

from tests import _path

# gitpython shells out to git at import, which is not needed here. Stub it.
sys.modules.setdefault("git", types.SimpleNamespace(Repo=object))
index_knowledge = _path.load_script("scripts/index-knowledge.py", "index_knowledge")


def flaky(errors, value="ok"):
    """A callable that raises each error in turn, then returns value."""
    pending = list(errors)

    def fn():
        if pending:
            raise pending.pop(0)
        return value
    return fn


class WithRetryTest(unittest.TestCase):
    def test_returns_after_dropped_connections(self):
        drop = httpx.RemoteProtocolError("Server disconnected without sending a response")
        with mock.patch.object(index_knowledge.time, "sleep") as sleep:
            got = index_knowledge._with_retry(flaky([drop, drop]), "embed")
        self.assertEqual(got, "ok")
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4])

    def test_timed_out_insert_is_not_retried(self):
        with mock.patch.object(index_knowledge.time, "sleep") as sleep:
            with self.assertRaises(ConnectionTimeout):
                index_knowledge._with_retry(
                    flaky([ConnectionTimeout("TIMEOUT", "read timed out", None)]), "index")
        sleep.assert_not_called()

    def test_gives_up_after_five_attempts(self):
        refused = OSConnectionError("N/A", "Connection refused", None)
        with mock.patch.object(index_knowledge.time, "sleep") as sleep:
            with self.assertRaises(OSConnectionError):
                index_knowledge._with_retry(flaky([refused] * 5), "index")
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4, 8, 16])

    def test_concurrent_retry_lines_stay_whole(self):
        drop = httpx.RemoteProtocolError("Server disconnected without sending a response")
        out = io.StringIO()
        with mock.patch.object(index_knowledge.time, "sleep"), \
                mock.patch.object(index_knowledge.sys, "stdout", out):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: index_knowledge._with_retry(flaky([drop] * 3), "embed"),
                              range(8)))
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 24)
        for line in lines:
            self.assertRegex(line, r"^    embed failed \(RemoteProtocolError\); "
                                   r"retrying in \d+s \(attempt \d/5\)$")


if __name__ == "__main__":
    unittest.main()
