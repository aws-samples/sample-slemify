"""eval/run_eval.py keeps the analyst's time to first token, so prefill on the
real question-plus-chunks prompt is readable from the evaluation instead of
from a separate query sent through the UI."""
import json
import unittest
from unittest import mock

import httpx

from tests import _path

run_eval = _path.load_script("eval/run_eval.py", "run_eval_for_ttft")

CASE = {"id": "c1", "query": "q", "expected_category": "", "must_include": ["x"]}


def sse(*events) -> bytes:
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n").encode()


def client_for(body: bytes):
    real = httpx.Client

    def handler(request):
        return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})

    return lambda **kw: real(transport=httpx.MockTransport(handler), **kw)


# One query that the gate flagged: the analyst drafts, the frontier model
# answers after it, and both report a first token.
ESCALATED = sse({"type": "steps", "steps": {}},
                {"type": "step_done", "name": "Triage (CPU)", "detail": "karpenter_config", "ms": 12},
                {"type": "step_done", "name": "Analyst SLM (CPU)", "detail": "time to first token", "ms": 31630},
                {"type": "token", "text": "a draft"},
                {"type": "answer_reset"},
                {"type": "step_done", "name": "LLM API (Bedrock escalation)",
                 "detail": "time to first token", "ms": 900},
                {"type": "token", "text": "an answer"},
                {"type": "cost", "usd": 0.02, "steps": {}, "total_ms": 41000},
                {"type": "total", "ms": 41000})
NO_ANALYST = sse({"type": "steps", "steps": {}},
                 {"type": "token", "text": "an answer"},
                 {"type": "cost", "usd": 0.0, "steps": {}, "total_ms": 9},
                 {"type": "total", "ms": 9})


class AnalystTtftTest(unittest.TestCase):
    def test_query_keeps_the_analyst_first_token_not_the_escalation(self):
        with mock.patch.object(run_eval.httpx, "Client", client_for(ESCALATED)):
            result = run_eval._query_once("q")
        self.assertEqual(result["analyst_ttft_ms"], 31630)

    def test_row_reports_the_median_over_the_runs(self):
        judge = mock.Mock(return_value={"verdict": "pass"})
        with mock.patch.object(run_eval.httpx, "Client", client_for(ESCALATED)), \
                mock.patch.object(run_eval, "judge", judge):
            row = run_eval.run_case(CASE, 2)
        self.assertEqual(row["analyst_ttft_ms_p50"], 31630)

    def test_a_query_with_no_analyst_step_reports_nothing(self):
        judge = mock.Mock(return_value={"verdict": "pass"})
        with mock.patch.object(run_eval.httpx, "Client", client_for(NO_ANALYST)), \
                mock.patch.object(run_eval, "judge", judge):
            row = run_eval.run_case(CASE, 1)
        self.assertIsNone(row["analyst_ttft_ms_p50"])


if __name__ == "__main__":
    unittest.main()
