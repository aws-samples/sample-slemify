"""server.py: /query opens with the step configuration and its cost event has
the query's total time; /config returns the step configuration."""
import contextlib
import io
import json
import unittest
from unittest import mock

from tests import _path  # noqa: F401

import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

STEP_KEYS = {"triage", "embed", "rerank", "analyst", "gate"}


class StubAgent:
    async def astream(self, state, stream_mode=None):
        yield {"type": "step_start", "name": "Embedder (hosted, Bedrock)", "note": "embedding query"}
        yield {"type": "step_done", "name": "Embedder (hosted, Bedrock)", "ms": 12, "detail": "x"}


class RaisingAgent:
    async def astream(self, state, stream_mode=None):
        yield {"type": "step_start", "name": "Embedder (hosted, Bedrock)", "note": "embedding query"}
        raise RuntimeError("boom")


def data_events(body: str) -> list:
    return [line[len("data: "):] for line in body.splitlines() if line.startswith("data: ")]


class ServerTest(unittest.TestCase):
    def setUp(self):
        # No context manager: the startup warmup (which calls the model pods)
        # does not run.
        self.client = TestClient(server.app)

    def test_query_stream(self):
        with mock.patch.object(server, "agent", StubAgent()):
            r = self.client.post("/query", json={"text": "hi"})
        self.assertEqual(r.status_code, 200)
        events = data_events(r.text)
        self.assertEqual(events[-1], "[DONE]")
        parsed = [json.loads(e) for e in events[:-1]]
        self.assertEqual(parsed[0]["type"], "steps")
        self.assertEqual(set(parsed[0]["steps"]), STEP_KEYS)
        self.assertEqual([e["type"] for e in parsed[1:3]], ["step_start", "step_done"])
        (cost,) = [e for e in parsed if e["type"] == "cost"]
        self.assertIsInstance(cost["total_ms"], int)
        (total,) = [e for e in parsed if e["type"] == "total"]
        self.assertEqual(total["ms"], cost["total_ms"])

    def test_query_failure_is_logged(self):
        out = io.StringIO()
        with mock.patch.object(server, "agent", RaisingAgent()), contextlib.redirect_stdout(out):
            r = self.client.post("/query", json={"text": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(data_events(r.text)[-1], "[DONE]")
        self.assertIn("Query failed: RuntimeError: boom", out.getvalue())

    def test_config_has_steps(self):
        with mock.patch.object(server.toolclient, "apply_enabled", lambda: False):
            r = self.client.get("/config")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIs(body["apply_enabled"], False)
        self.assertEqual(set(body["steps"]), STEP_KEYS)


if __name__ == "__main__":
    unittest.main()
