"""agent/tools.py: investigate_cluster returns the NodePool summaries next to
the unhealthy pods when a pending pod picks its node by label, so the pool's
limits are in the same evidence as the pod events. The live-evidence question
("why are the payments-api pods in my cluster stuck Pending?") got the label
and the taint and nothing about spec.limits.cpu. No test calls the API server.
"""
import types
import unittest
from unittest import mock

from tests import _path  # noqa: F401

from agent import tools  # noqa: E402

# The pool demo/scenarios/limited-nodepool.yaml creates: it carries the label
# the pods select and a cap that provisions nothing.
LIMITED_POOL = {
    "metadata": {"name": "demo-payments"},
    "spec": {
        "limits": {"cpu": "0"},
        "template": {"spec": {"requirements": [
            {"key": "karpenter.sh/capacity-type", "values": ["on-demand"]},
            {"key": "kubernetes.io/arch", "values": ["arm64"]},
        ]}},
        "disruption": {"consolidationPolicy": "WhenEmptyOrUnderutilized"},
    },
}
QUERY = "why are the payments-api pods in my cluster stuck Pending? check the nodepool they target"


def pod(name, node_selector=None, phase="Pending"):
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(name=name, namespace="default"),
        spec=types.SimpleNamespace(node_selector=node_selector, affinity=None, tolerations=None),
        status=types.SimpleNamespace(phase=phase, container_statuses=None))


def investigate(pods, query=QUERY):
    """Run investigate_cluster against a fixed pod list and one NodePool."""
    api = mock.Mock()
    api.list_pod_for_all_namespaces.return_value = types.SimpleNamespace(items=pods)
    with mock.patch.object(tools, "_k8s_available", True), \
            mock.patch.object(tools.k8s_client, "CoreV1Api", lambda: api), \
            mock.patch.object(tools, "_latest_warning_event",
                              lambda *a: "FailedScheduling: no nodes available"), \
            mock.patch.object(tools, "_nodepool_summaries",
                              lambda: tools._format_nodepool(LIMITED_POOL)):
        return tools.investigate_cluster({"query": query})


class ProblemPodEvidenceTest(unittest.TestCase):
    def test_pending_pod_with_a_node_label_gets_the_pool_limits(self):
        out = investigate([pod("payments-api-1", {"demo.slemify.io/pool": "payments"})])
        self.assertIn("PROBLEM PODS:", out)
        self.assertIn("payments-api-1", out)
        self.assertIn("NODEPOOLS:", out)
        self.assertIn("limits=cpu=0", out)

    def test_unhealthy_pod_without_a_node_label_keeps_the_short_answer(self):
        out = investigate([pod("payments-api-1")])
        self.assertIn("PROBLEM PODS:", out)
        self.assertNotIn("NODEPOOLS:", out)

    def test_healthy_cluster_still_lists_the_pools(self):
        out = investigate([pod("payments-api-1", phase="Succeeded")])
        self.assertIn("No pending or unhealthy pods found", out)
        self.assertIn("limits=cpu=0", out)

    def test_a_named_pool_is_not_listed_twice(self):
        api = mock.Mock()
        api.list_pod_for_all_namespaces.return_value = types.SimpleNamespace(
            items=[pod("payments-api-1", {"demo.slemify.io/pool": "payments"})])
        custom = mock.Mock()
        custom.get_cluster_custom_object.return_value = LIMITED_POOL
        with mock.patch.object(tools, "_k8s_available", True), \
                mock.patch.object(tools.k8s_client, "CoreV1Api", lambda: api), \
                mock.patch.object(tools.k8s_client, "CustomObjectsApi", lambda: custom), \
                mock.patch.object(tools, "_latest_warning_event", lambda *a: None), \
                mock.patch.object(tools, "_nodepool_summaries",
                                  lambda: tools._format_nodepool(LIMITED_POOL)):
            out = tools.investigate_cluster(
                {"query": "check the pods on the nodepool `demo-payments` in my cluster"})
        self.assertIn("NODEPOOL:", out)
        self.assertNotIn("NODEPOOLS:", out)
        self.assertEqual(out.count("limits=cpu=0"), 1)


if __name__ == "__main__":
    unittest.main()
