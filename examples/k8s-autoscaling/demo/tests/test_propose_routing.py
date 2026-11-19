"""agent/graph.py: a flagged draft goes to propose only when the question asks
about the user's own resources. A flagged documentation question escalates.

The questions are the ones the workshop sends: DOC_QUESTIONS are the cases in
eval/cases.yaml, LIVE_QUESTIONS are from the live-evidence page. In the
end-to-end run every eval question that the gate flagged ended in propose,
because the old test matched the substring "scal". No test calls a model,
OpenSearch or the cluster."""
import asyncio
import unittest
from unittest import mock

from tests import _path  # noqa: F401

from agent import config, extract, graph  # noqa: E402

DOC_QUESTIONS = [
    "for a karpenter v1 NodePool, what are the valid values for "
    "spec.disruption.consolidationPolicy and what does each mean?",
    "does karpenter handle EC2 Spot interruption notices automatically, or do I need to "
    "install Node Termination Handler separately on a current version?",
    "in KEDA, how do I let a workload scale to zero when there are no events?",
    "for a CPU-based autoscaling trigger, what metricType values are valid?",
    "What does minValues do in a Karpenter NodePool requirement, and what happens if it "
    "cannot be satisfied?",
    "what's the maximum number of NodePools AWS officially recommends for a single "
    "5000-node EKS cluster, with exact numbers?",
]
LIVE_QUESTIONS = [
    "why are the payments-api pods in my cluster stuck Pending? check the nodepool they target",
    "my nodes are not scaling up, check the cluster",
    "describe the nodepool demo-payments in my cluster",
]

SLM_DRAFT = "A draft the gate will flag."
LLM_ANSWER = "The escalated answer."


async def _tokens(text):
    yield text


async def _stub_retrieve(state):
    return {"docs": ["Multiple NodePools may point to the same EC2NodeClass."]}


def _run(query):
    """Run the graph (TRIAGE=off, GATE=llm) with the cluster tools reachable and
    a gate that flags the first draft and accepts the second. Returns the
    recorded outcomes and the stream events."""
    outcomes, checks = [], []

    def fake_gate(q, draft, context):
        checks.append(draft)
        return len(checks) == 1, "P(unsupported) 0.74"

    with mock.patch.object(config, "TRIAGE", "off"), \
            mock.patch.object(config, "GATE", "llm"), \
            mock.patch.object(config, "ANALYST", "slm"), \
            mock.patch.object(graph, "n_retrieve", _stub_retrieve), \
            mock.patch.object(graph.generation, "stream_slm", lambda q, c: _tokens(SLM_DRAFT)), \
            mock.patch.object(graph.generation, "stream_llm", lambda q, c: _tokens(LLM_ANSWER)), \
            mock.patch.object(graph.generation, "stream_calibrated", lambda q, c, r: _tokens("calibrated")), \
            mock.patch.object(graph.gate, "check", fake_gate), \
            mock.patch.object(graph, "validate_draft_fix", lambda d: []), \
            mock.patch.object(graph.toolclient, "available", lambda: True), \
            mock.patch.object(graph.toolclient, "detect_remediation", lambda q: None), \
            mock.patch.object(graph.toolclient, "named_target", lambda q: None), \
            mock.patch.object(graph.metrics, "record_gate_check", lambda *a: None), \
            mock.patch.object(graph.metrics, "record_outcome",
                              lambda category, outcome: outcomes.append(outcome)):
        agent = graph.build_agent()

        async def go():
            return [e async for e in agent.astream({"query": query, "autopilot": False},
                                                   stream_mode="custom")]

        events = asyncio.run(go())
    return outcomes, events


class RoutingTestTest(unittest.TestCase):
    """The test the gate applies: an inspect verb and the user's own resources."""

    def test_documentation_questions_do_not_ask_for_cluster_action(self):
        for q in DOC_QUESTIONS:
            self.assertFalse(extract.wants_cluster_action(q), q)

    def test_live_questions_ask_for_cluster_action(self):
        for q in LIVE_QUESTIONS:
            self.assertTrue(extract.wants_cluster_action(q), q)

    def test_most_documentation_questions_still_read_as_operational(self):
        # Why the old condition fired: the operational test is a substring
        # match, so "autoscaling" and "scale to zero" match it.
        self.assertTrue(any(extract.is_operational(q) for q in DOC_QUESTIONS))


class ProposeRoutingTest(unittest.TestCase):
    def test_flagged_documentation_question_escalates(self):
        for q in DOC_QUESTIONS:
            outcomes, events = _run(q)
            self.assertEqual(outcomes, ["llm_pass"], q)
            self.assertIn("LLM API (Bedrock escalation)", [e.get("name") for e in events], q)

    def test_flagged_live_question_proposes_the_read_only_check(self):
        for q in LIVE_QUESTIONS:
            outcomes, events = _run(q)
            self.assertEqual(outcomes, ["propose"], q)
            self.assertNotIn("LLM API (Bedrock escalation)", [e.get("name") for e in events], q)


if __name__ == "__main__":
    unittest.main()
