"""agent/graph.py: after the gate rejects the analyst's draft, the frontier
model's escalated answer is what the gate checks next. No test calls a model,
OpenSearch or the cluster."""
import asyncio
import unittest
from unittest import mock

from tests import _path  # noqa: F401
from agent import config, graph  # noqa: E402

SLM_DRAFT = "The system allows an unlimited number of NodePools."
LLM_ANSWER = "The documentation gives no maximum."


async def _tokens(text):
    yield text


async def _stub_retrieve(state):
    return {"docs": ["Multiple NodePools may point to the same EC2NodeClass."]}


def _run(verdicts):
    """Run the graph (TRIAGE=off, GATE=llm) with a gate that returns the given
    escalate flags in order. Returns (drafts the gate saw, stream events)."""
    seen = []

    def fake_gate(query, draft, context):
        seen.append(draft)
        return verdicts[len(seen) - 1], "reason"

    with mock.patch.object(config, "TRIAGE", "off"), \
            mock.patch.object(config, "GATE", "llm"), \
            mock.patch.object(config, "ANALYST", "slm"), \
            mock.patch.object(graph, "n_retrieve", _stub_retrieve), \
            mock.patch.object(graph.generation, "stream_slm", lambda q, c: _tokens(SLM_DRAFT)), \
            mock.patch.object(graph.generation, "stream_llm", lambda q, c: _tokens(LLM_ANSWER)), \
            mock.patch.object(graph.generation, "stream_calibrated", lambda q, c, r: _tokens("calibrated")), \
            mock.patch.object(graph.gate, "check", fake_gate), \
            mock.patch.object(graph, "validate_draft_fix", lambda d: []), \
            mock.patch.object(graph.toolclient, "detect_remediation", lambda q: None), \
            mock.patch.object(graph.toolclient, "named_target", lambda q: None), \
            mock.patch.object(graph.metrics, "record_gate_check", lambda *a: None), \
            mock.patch.object(graph.metrics, "record_outcome", lambda *a: None):
        agent = graph.build_agent()

        async def go():
            return [e async for e in agent.astream({"query": "q", "autopilot": False},
                                                   stream_mode="custom")]
        events = asyncio.run(go())
    return seen, events


class EscalationTest(unittest.TestCase):
    def test_second_gate_check_reads_the_escalated_answer(self):
        seen, events = _run([True, False])
        self.assertEqual(seen, [SLM_DRAFT, LLM_ANSWER])
        self.assertNotIn("Calibrated answer (LLM)", [e.get("name") for e in events])

    def test_rejected_escalated_answer_abstains(self):
        seen, events = _run([True, True])
        self.assertEqual(seen, [SLM_DRAFT, LLM_ANSWER])
        self.assertIn("Calibrated answer (LLM)", [e.get("name") for e in events])


if __name__ == "__main__":
    unittest.main()
