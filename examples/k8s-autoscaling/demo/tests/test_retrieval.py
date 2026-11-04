"""agent/retrieval.py: the fused fallback order, and rerank_docs falling back
to it when the reranker fails. No test calls OpenSearch or the reranker."""
import unittest
from unittest import mock

import httpx

from tests import _path  # noqa: F401

from agent import config, retrieval  # noqa: E402


def hits(*texts):
    return {"hits": {"hits": [{"_source": {"text": t, "source": "s", "section": t}} for t in texts]}}


def fmt(t):
    return f"[s / {t}]\n{t}"


class FuseRankedTest(unittest.TestCase):
    def test_top_hit_of_second_list_stays_near_the_top(self):
        dense = [f"d{i}" for i in range(30)]
        fused = retrieval.fuse_ranked(dense, ["lex0", "lex1"])
        self.assertIn("lex0", fused[:3])

    def test_doc_in_both_lists_ranks_above_docs_in_one(self):
        fused = retrieval.fuse_ranked(["a", "b", "c"], ["c", "x"])
        self.assertEqual(fused[0], "c")

    def test_ties_keep_first_list_order(self):
        fused = retrieval.fuse_ranked(["a", "b"], ["x", "y"])
        self.assertEqual(fused, ["a", "x", "b", "y"])

    def test_duplicates_appear_once(self):
        fused = retrieval.fuse_ranked(["a", "b", "a"], ["b", "a"])
        self.assertEqual(sorted(fused), ["a", "b"])


class FailingClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, *args, **kwargs):
        raise httpx.ReadTimeout("timed out")


class RerankDocsTest(unittest.TestCase):
    docs = ["d0", "d1", "d2", "d3"]
    fallback = ["d0", "lex0", "d1", "d2"]

    def test_failure_uses_fallback_when_given(self):
        with mock.patch.object(config, "RERANK", "on"), \
                mock.patch.object(retrieval.httpx, "Client", FailingClient):
            got = retrieval.rerank_docs("q", self.docs, 2, self.fallback)
        self.assertEqual(got, ["d0", "lex0"])

    def test_failure_uses_docs_without_fallback(self):
        with mock.patch.object(config, "RERANK", "on"), \
                mock.patch.object(retrieval.httpx, "Client", FailingClient):
            got = retrieval.rerank_docs("q", self.docs, 2)
        self.assertEqual(got, ["d0", "d1"])

    def test_rerank_off_ignores_fallback(self):
        with mock.patch.object(config, "RERANK", "off"):
            got = retrieval.rerank_docs("q", self.docs, 2, self.fallback)
        self.assertEqual(got, ["d0", "d1"])


class HybridCandidatesTest(unittest.TestCase):
    def test_returns_dense_first_union_and_fused_order(self):
        dense = [f"d{i}" for i in range(6)]
        search = mock.Mock(side_effect=[hits(*dense), hits("lex0", "d3")])
        with mock.patch.object(config, "opensearch", mock.Mock(search=search)):
            union, fused = retrieval.hybrid_candidates([0.0], "query")
        self.assertEqual(union, [fmt(t) for t in dense + ["lex0"]])
        self.assertEqual(sorted(fused), sorted(union))
        # d3 is in both lists, lex0 is the top lexical hit.
        self.assertEqual(fused[0], fmt("d3"))
        self.assertIn(fmt("lex0"), fused[:3])


if __name__ == "__main__":
    unittest.main()
