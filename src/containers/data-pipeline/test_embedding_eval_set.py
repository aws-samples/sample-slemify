"""What the retriever pipeline writes as the recall inputs.

Recall@k is measured over the held-out pairs in eval.jsonl against the chunks
in corpus.jsonl. These tests run the embedding pipeline twice with two
different model backends, which is what two accounts have, and compare those
two files. With committed held-out pairs (data.evaluation.labeled and
evaluation.pairs at 0) both files are identical between the runs. With
generated held-out pairs they are not, and the second test records that.

Run from this directory: python3 -m unittest test_embedding_eval_set -v
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

import main  # noqa: E402

PATH = "k8s-autoscaling/data/"
QUERIES = [{"source": f"{PATH}queries/doc-{i}.txt",
            "content": f"Document {i}. " + ("filler words " * 40)}
           for i in range(12)]
EVAL_DOCS = [{"source": f"{PATH}eval-queries/eval-{i}.txt",
              "content": f"Held-out document {i}. " + ("other words " * 40)}
             for i in range(4)]
# Three committed held-out pairs. Each positive is the text of a chunk the
# corpus already holds, so the retrieval index can rank it.
COMMITTED = [{"query": "why do pods stay Pending when a NodePool has no capacity",
              "positive": QUERIES[0]["content"].strip()},
             {"query": "how do I stop consolidation from replacing nodes every minute",
              "positive": QUERIES[5]["content"].strip()},
             {"query": "which field caps the CPU a NodePool may provision",
              "positive": QUERIES[9]["content"].strip()}]
COMMITTED_FILE = {"source": f"{PATH}eval-labeled/retriever-heldout.jsonl",
                  "content": "\n".join(json.dumps(r) for r in COMMITTED)}


def data_cfg(eval_pairs, labeled):
    cfg = {"bucket": "slemify-data", "path": PATH,
           "sources": [{"path": "queries/", "type": "raw"}],
           "evaluation": {"pairs": eval_pairs, "model": "a-frontier-model",
                          "sources": [{"path": "eval-queries/", "type": "raw"}]}}
    if labeled:
        cfg["evaluation"]["labeled"] = [{"path": "eval-labeled/retriever-heldout.jsonl"}]
    return cfg


def run(account, eval_pairs=0, labeled=True):
    """Run the embedding pipeline for one account. The questions a model call
    returns differ per account, in wording and in how many come back, which is
    the behavior a seeded chunk sample does not change."""
    written = {}

    def read_raw_sources(bucket, path, sources):
        out = []
        for source in sources:
            prefix = source.get("path", "")
            if prefix.startswith("eval-labeled/"):
                out.append(COMMITTED_FILE)
            elif prefix.startswith("eval-queries/"):
                out.extend(EVAL_DOCS)
            else:
                out.extend(QUERIES)
        return out

    def queries(backend, domain, chunk_text, n_queries):
        n = 1 + (len(chunk_text) + len(account)) % 2
        return [f"{account} question {i} about {chunk_text[:14]}" for i in range(n)]

    with mock.patch.object(main, "_select_backend", lambda model, endpoint: object()), \
            mock.patch.object(main, "_calculate_concurrency", lambda model: 8), \
            mock.patch.object(main, "_generate_queries_for_chunk", queries), \
            mock.patch.object(main, "read_raw_sources", read_raw_sources), \
            mock.patch.object(main, "write_jsonl_to_s3",
                              lambda bucket, key, records: written.__setitem__(key, records)):
        main.run_embedding_pipeline({}, data_cfg(eval_pairs, labeled), "proj", "d",
                                    QUERIES, {"model": "a-frontier-model", "pairs": 30})
    return written


class CommittedHeldOutSetTest(unittest.TestCase):
    def setUp(self):
        self.first = run("account-one")
        self.second = run("account-two")

    def test_two_accounts_write_the_same_held_out_pairs(self):
        self.assertEqual(self.first["proj/processed/eval.jsonl"],
                         self.second["proj/processed/eval.jsonl"])

    def test_the_held_out_pairs_are_the_committed_ones_and_nothing_else(self):
        held_out = self.first["proj/processed/eval.jsonl"]
        self.assertEqual([r["query"] for r in held_out], [r["query"] for r in COMMITTED])
        self.assertEqual([r["positive"] for r in held_out], [r["positive"] for r in COMMITTED])
        self.assertEqual({r["origin"] for r in held_out}, {"real"})

    def test_two_accounts_write_the_same_retrieval_corpus(self):
        self.assertEqual(self.first["proj/processed/corpus.jsonl"],
                         self.second["proj/processed/corpus.jsonl"])

    def test_the_training_pairs_still_differ_between_accounts(self):
        # The limit of this change: the questions trained on are written per
        # account, so the tuned encoder is not the same encoder.
        first = [r["query"] for r in self.first["proj/processed/train.jsonl"]]
        second = [r["query"] for r in self.second["proj/processed/train.jsonl"]]
        self.assertNotEqual(first, second)
        self.assertTrue(all(q.startswith("account-one") for q in first))


class GeneratedHeldOutSetTest(unittest.TestCase):
    """Without committed pairs the held-out questions come from a model call
    per chunk, so the two accounts measure recall on different questions."""

    def test_generated_held_out_pairs_differ_between_accounts(self):
        first = run("account-one", eval_pairs=12, labeled=False)
        second = run("account-two", eval_pairs=12, labeled=False)
        self.assertNotEqual([r["query"] for r in first["proj/processed/eval.jsonl"]],
                            [r["query"] for r in second["proj/processed/eval.jsonl"]])

    def test_committed_pairs_are_appended_when_generation_stays_on(self):
        # Today's retriever config: a generated held-out set with committed
        # pairs added to it. The committed part is comparable, the rest is not.
        held_out = run("account-one", eval_pairs=12)["proj/processed/eval.jsonl"]
        self.assertEqual({r["origin"] for r in held_out}, {"synthetic", "real"})
        self.assertGreater(len(held_out), len(COMMITTED))


if __name__ == "__main__":
    unittest.main()
