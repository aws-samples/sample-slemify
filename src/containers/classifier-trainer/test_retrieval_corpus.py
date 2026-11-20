"""The document list recall@k is scored over, and the order it holds.

recall@k is measured over the held-out pairs against one list of documents:
the training chunks plus any held-out answer document they do not already
hold. A tie in similarity is broken by position in that list, so the order
decides which document lands inside the top k. Building the list by iterating
a Python set made the order depend on the process hash seed, which moved
recall@2 between runs of the same data.

These tests check the order directly, and then run the metric itself in
subprocesses with different hash seeds and compare the numbers.

Run from this directory: python3 -m unittest test_retrieval_corpus -v
"""
import json
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

import train  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = ("0", "1", "2", "3", "4", "5", "6", "7")

# Two held-out documents and a training chunk that all look alike to the
# encoder below, so every ranking is a tie and position decides it. DOC_A
# answers two of the three questions and DOC_B answers one, so recall@2 is
# 2/3 when DOC_A comes first in the corpus and 1/3 when DOC_B does.
TRAIN_CHUNK = "a training chunk about autoscaling"
DOC_A = "held-out document a about autoscaling"
DOC_B = "held-out document b about autoscaling"
EVAL_ROWS = [{"query": "first question about scaling up", "positive": DOC_A},
             {"query": "second question about scaling up", "positive": DOC_A},
             {"query": "third question about scaling up", "positive": DOC_B}]

# Run in a fresh interpreter so PYTHONHASHSEED takes effect.
PROGRAM = f"""
import json, sys
import numpy as np
sys.path.insert(0, {HERE!r})
import train


class TiedEncoder:
    "Every text gets the same vector, so every similarity ties."

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32) / 2.0


eval_rows = {EVAL_ROWS!r}
corpus = train._effective_corpus([{TRAIN_CHUNK!r}], eval_rows)
metrics = train._retrieval_metrics(TiedEncoder(), eval_rows, corpus, ks=(1, 2))
print(json.dumps({{"corpus": corpus, "recall@2": metrics["recall@2"]}}))
"""


def run_with_seed(seed):
    env = dict(os.environ, PYTHONHASHSEED=seed, AWS_DEFAULT_REGION="us-west-2")
    out = subprocess.run([sys.executable, "-c", PROGRAM], check=True, env=env,
                         capture_output=True, text=True)
    return json.loads(out.stdout)


class EffectiveCorpusOrderTest(unittest.TestCase):
    def test_held_out_documents_are_appended_in_record_order(self):
        self.assertEqual(train._effective_corpus([TRAIN_CHUNK], EVAL_ROWS),
                         [TRAIN_CHUNK, DOC_A, DOC_B])

    def test_a_document_two_pairs_share_is_added_once(self):
        corpus = train._effective_corpus([TRAIN_CHUNK], EVAL_ROWS)
        self.assertEqual(corpus.count(DOC_A), 1)

    def test_a_document_the_training_corpus_already_holds_is_not_repeated(self):
        self.assertEqual(train._effective_corpus([DOC_A, TRAIN_CHUNK], EVAL_ROWS),
                         [DOC_A, TRAIN_CHUNK, DOC_B])

    def test_the_training_corpus_is_not_modified(self):
        training = [TRAIN_CHUNK]
        train._effective_corpus(training, EVAL_ROWS)
        self.assertEqual(training, [TRAIN_CHUNK])


class HashSeedTest(unittest.TestCase):
    """Eight interpreters with eight hash seeds. Set iteration order changes
    with the seed, so the old corpus build gave two different answers here."""

    @classmethod
    def setUpClass(cls):
        cls.results = [run_with_seed(seed) for seed in SEEDS]

    def test_the_corpus_is_the_same_list_under_every_hash_seed(self):
        orders = {tuple(r["corpus"]) for r in self.results}
        self.assertEqual(orders, {(TRAIN_CHUNK, DOC_A, DOC_B)})

    def test_recall_at_2_is_the_same_number_under_every_hash_seed(self):
        values = {r["recall@2"] for r in self.results}
        self.assertEqual(len(values), 1, f"recall@2 moved with the hash seed: {values}")
        self.assertAlmostEqual(values.pop(), 2 / 3)


if __name__ == "__main__":
    unittest.main()
