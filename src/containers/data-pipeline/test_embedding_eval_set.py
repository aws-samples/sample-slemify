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


# The shipped file has the same shape: 14 answer documents, three questions
# each, 42 pairs in all, grouped by document. Each record states its own
# origin, and that origin is "synthetic", because a model wrote the questions.
DOCS = ([d["content"].strip() for d in QUERIES] + [d["content"].strip() for d in EVAL_DOCS])[:14]
PAIRS_PER_DOC = 3


def committed(missing=0):
    """The committed held-out pairs. Each positive is the text of a chunk one
    of the sources holds, so the retrieval index can rank it. With missing set,
    the first that many positives are replaced by text no source holds, as an
    edited document would leave. The pairs are grouped by document, so missing
    at PAIRS_PER_DOC is one edited document."""
    pairs = [{"query": f"committed question {q} about document {d}",
              "positive": doc, "origin": "synthetic"}
             for d, doc in enumerate(DOCS) for q in range(PAIRS_PER_DOC)]
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

    def test_the_origin_each_record_states_is_the_one_that_is_written(self):
        # The shipped pairs declare "synthetic", because a model wrote the
        # questions. The pipeline must not restamp them "real": the report
        # scores the two origins separately.
        held_out = self.first["proj/processed/eval.jsonl"]
        self.assertEqual({r["origin"] for r in held_out}, {"synthetic"})

    def test_the_log_names_the_origin_the_records_carry(self):
        with self.assertLogs(main.logger, level="INFO") as logs:
            run("account-one")
        self.assertTrue(any("committed eval pairs (origin=synthetic)" in m for m in logs.output),
                        f"the log did not name the shipped origin: {logs.output}")

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
    an answer it cannot find to the index rather than dropping it. So a pair
    whose document changed has to be reported here, where both the pairs and
    the fresh chunks are in hand.

    The limit counts pairs, not documents: it is a share of the committed
    pairs. Because each document answers three of them, one edited document
    takes three pairs past the limit and fails the stage."""

    def test_pairs_that_still_match_are_reported_as_matching(self):
        with self.assertLogs(main.logger, level="INFO") as logs:
            run("account-one")
        self.assertTrue(any("All 42 committed eval pairs match" in m for m in logs.output))

    def test_one_unmatched_pair_warns_and_the_run_goes_on(self):
        # 1 of 42 is 2.4%, under the 5% limit. A single pair can lose its
        # document to a reworded heading, which is worth a loud count and no
        # more.
        with self.assertLogs(main.logger, level="WARNING") as logs:
            written = run("account-one", missing=1)
        self.assertTrue(any("1 of 42 committed eval pairs (2.4%)" in m for m in logs.output))
        self.assertTrue(any("no matching chunk" in m for m in logs.output))
        self.assertEqual(len(written["proj/processed/eval.jsonl"]), 42)

    def test_one_edited_document_takes_its_three_pairs_and_fails_the_stage(self):
        # The pairs are grouped by document, so the first three are the three
        # questions one document answers. 3 of 42 is 7.1%, above the limit, so
        # the run stops before training measures recall that cannot be compared
        # with any other run.
        with self.assertLogs(main.logger, level="ERROR") as logs:
            with self.assertRaises(SystemExit) as exit_code:
                run("account-one", missing=PAIRS_PER_DOC)
        self.assertEqual(exit_code.exception.code, 1)
        self.assertTrue(any("3 of 42 committed eval pairs (7.1%)" in m for m in logs.output))
        self.assertTrue(any("Regenerate the committed pairs" in m for m in logs.output))

    def test_the_limit_is_a_share_of_the_pairs(self):
        self.assertEqual(main.COMMITTED_PAIR_DRIFT_LIMIT, 0.05)
        # One document's pairs are above that share for the shipped set, which
        # is why one edited document fails rather than warns.
        self.assertGreater(PAIRS_PER_DOC / len(COMMITTED), main.COMMITTED_PAIR_DRIFT_LIMIT)


class GeneratedHeldOutSetTest(unittest.TestCase):
    """Without committed pairs the held-out questions come from a model call
    per chunk, so the two accounts measure recall on different questions."""

    def test_generated_held_out_pairs_differ_between_accounts(self):
        first = run("account-one", eval_pairs=12, labeled=False)
        second = run("account-two", eval_pairs=12, labeled=False)
        self.assertNotEqual([r["query"] for r in first["proj/processed/eval.jsonl"]],
                            [r["query"] for r in second["proj/processed/eval.jsonl"]])

    def test_committed_pairs_are_appended_when_generation_stays_on(self):
        # The other arrangement: a generated held-out set with the committed
        # pairs added to it. The committed part is comparable, the rest is not.
        held_out = run("account-one", eval_pairs=12)["proj/processed/eval.jsonl"]
        self.assertGreater(len(held_out), len(COMMITTED))
        self.assertEqual([r["query"] for r in held_out[-len(COMMITTED):]],
                         [r["query"] for r in COMMITTED])
        # Both parts were written by a model, so both read "synthetic".
        self.assertEqual({r["origin"] for r in held_out}, {"synthetic"})


if __name__ == "__main__":
    unittest.main()
