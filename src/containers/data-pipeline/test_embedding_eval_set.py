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
# The text of a document no source holds, for the drift cases below.
GONE = "A document that was edited away. " + ("other words " * 40)


def committed(missing=0):
    """Forty committed held-out pairs. Each positive is the text of a chunk one
    of the sources holds, half of them from the held-out documents, so the
    retrieval index can rank it. With missing set, that many positives are
    replaced by text no source holds, as an edited document would leave."""
    docs = [d["content"].strip() for d in QUERIES] + [d["content"].strip() for d in EVAL_DOCS]
    pairs = [{"query": f"committed question {i} about autoscaling",
              "positive": docs[i % len(docs)]} for i in range(40)]
    for i in range(missing):
        pairs[i]["positive"] = GONE
    return pairs


COMMITTED = committed()


def committed_file(missing=0):
    return {"source": f"{PATH}eval-labeled/retriever-heldout.jsonl",
            "content": "\n".join(json.dumps(r) for r in committed(missing))}


def data_cfg(eval_pairs, labeled):
    cfg = {"bucket": "slemify-data", "path": PATH,
           "sources": [{"path": "queries/", "type": "raw"}],
           "evaluation": {"pairs": eval_pairs, "model": "a-frontier-model",
                          "sources": [{"path": "eval-queries/", "type": "raw"}]}}
    if labeled:
        cfg["evaluation"]["labeled"] = [{"path": "eval-labeled/retriever-heldout.jsonl"}]
    return cfg


def run(account, eval_pairs=0, labeled=True, missing=0):
    """Run the embedding pipeline for one account. The questions a model call
    returns differ per account, in wording and in how many come back, which is
    the behavior a seeded chunk sample does not change."""
    written = {}

    def read_raw_sources(bucket, path, sources):
        out = []
        for source in sources:
            prefix = source.get("path", "")
            if prefix.startswith("eval-labeled/"):
                out.append(committed_file(missing))
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


class CorpusDriftTest(unittest.TestCase):
    """A committed pair names its answer document by text, and the trainer adds
    an answer it cannot find to the index rather than dropping it. So an edited
    document has to be reported here, where both the pairs and the fresh chunks
    are in hand."""

    def test_pairs_that_still_match_are_reported_as_matching(self):
        with self.assertLogs(main.logger, level="INFO") as logs:
            run("account-one")
        self.assertTrue(any("All 40 committed eval pairs match" in m for m in logs.output))

    def test_a_few_edited_documents_warn_and_the_run_goes_on(self):
        # 1 of 40 is 2.5%, under the 5% limit.
        with self.assertLogs(main.logger, level="WARNING") as logs:
            written = run("account-one", missing=1)
        self.assertTrue(any("1 of 40 committed eval pairs (2.5%)" in m for m in logs.output))
        self.assertTrue(any("no matching chunk" in m for m in logs.output))
        self.assertEqual(len(written["proj/processed/eval.jsonl"]), 40)

    def test_many_edited_documents_fail_the_stage(self):
        # 4 of 40 is 10%, above the limit, so the run stops before training
        # spends time measuring recall that cannot be compared with anything.
        with self.assertLogs(main.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exit_code:
                run("account-one", missing=4)
        self.assertEqual(exit_code.exception.code, 1)
        self.assertTrue(any("4 of 40 committed eval pairs (10.0%)" in m for m in logs.output))
        self.assertTrue(any("Regenerate the committed pairs" in m for m in logs.output))

    def test_the_limit_is_the_documented_one(self):
        self.assertEqual(main.COMMITTED_PAIR_DRIFT_LIMIT, 0.05)


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
