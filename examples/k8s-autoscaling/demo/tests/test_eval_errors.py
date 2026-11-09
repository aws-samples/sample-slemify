"""eval/run_eval.py: an `error` event from /query scores the run as ERR
(not FAIL), and a normal stream is still collected and judged."""
import json
import unittest
from unittest import mock

import httpx

from tests import _path

run_eval = _path.load_script("eval/run_eval.py", "run_eval_for_errors")

CASE = {"id": "c1", "query": "q", "expected_category": "", "must_include": ["x"]}


def sse(*events) -> bytes:
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n").encode()


def client_for(body: bytes):
    """An httpx.Client factory whose requests all return `body` as SSE."""
    real = httpx.Client

    def handler(request):
        return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})

    return lambda **kw: real(transport=httpx.MockTransport(handler), **kw)


FAILED = sse({"type": "steps", "steps": {}},
             {"type": "step_start", "name": "Analyst SLM (CPU)", "note": ""},
             {"type": "error", "message": "RuntimeError: boom"},
             {"type": "cost", "usd": 0.0, "steps": {}, "total_ms": 5},
             {"type": "total", "ms": 5})
OK = sse({"type": "steps", "steps": {}},
         {"type": "token", "text": "an answer"},
         {"type": "cost", "usd": 0.01, "steps": {}, "total_ms": 9},
         {"type": "total", "ms": 9})


class EvalErrorTest(unittest.TestCase):
    def test_error_event_raises(self):
        with mock.patch.object(run_eval.httpx, "Client", client_for(FAILED)):
            with self.assertRaises(run_eval.AgentError) as cm:
                run_eval._query_once("q")
        self.assertEqual(str(cm.exception), "RuntimeError: boom")

    def test_error_event_scores_err_without_judging(self):
        judge = mock.Mock()
        with mock.patch.object(run_eval.httpx, "Client", client_for(FAILED)), \
                mock.patch.object(run_eval, "judge", judge):
            row = run_eval.run_case(CASE, 2)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["runs"], ["error", "error"])
        self.assertEqual(row["errors"], 2)
        self.assertIn("RuntimeError: boom", row["judge"]["reason"])
        judge.assert_not_called()

    def test_normal_stream_is_judged(self):
        judge = mock.Mock(return_value={"verdict": "pass"})
        with mock.patch.object(run_eval.httpx, "Client", client_for(OK)), \
                mock.patch.object(run_eval, "judge", judge):
            row = run_eval.run_case(CASE, 1)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["answer"], "an answer")
        judge.assert_called_once()


if __name__ == "__main__":
    unittest.main()
