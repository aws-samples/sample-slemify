"""The committed retriever held-out set still matches the committed documents.

`examples/k8s-autoscaling/data/eval-labeled/retriever-heldout.jsonl` names each
pair's answer document by its exact chunk text, and the trainer looks it up that
way. Editing a document under `data/eval-queries/` therefore breaks the pairs
that point at it. The pipeline catches that at run time; this test catches it
before anyone deploys. Run from this directory:
python3 -m unittest test_committed_pairs_file -v
"""
import collections
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

import main  # noqa: E402

EXAMPLE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "examples", "k8s-autoscaling"))
PAIRS = os.path.join(EXAMPLE, "data", "eval-labeled", "retriever-heldout.jsonl")
DOCS = os.path.join(EXAMPLE, "data", "eval-queries")


@unittest.skipUnless(os.path.exists(PAIRS), "example data is not in this tree")
class CommittedPairsFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PAIRS, encoding="utf-8") as fh:
            cls.pairs = [json.loads(line) for line in fh if line.strip()]
        raw = []
        for name in sorted(os.listdir(DOCS)):
            with open(os.path.join(DOCS, name), encoding="utf-8") as fh:
                raw.append({"source": name, "content": fh.read().strip()})
        cls.chunks = {c["text"] for c in main._chunk_documents(raw)}

    def test_every_pair_points_at_a_current_chunk(self):
        missing = [p["query"] for p in self.pairs if p["positive"] not in self.chunks]
        self.assertEqual(missing, [], "regenerate the committed pairs: these have no chunk")

    def test_the_records_have_the_fields_the_pipeline_reads(self):
        for pair in self.pairs:
            self.assertTrue(pair.get("query"))
            self.assertTrue(pair.get("positive"))
            # A model wrote these questions, so they must not claim to be the
            # human-written set the report scores separately.
            self.assertEqual(pair.get("origin"), "synthetic")

    def test_no_question_is_asked_twice(self):
        queries = [p["query"] for p in self.pairs]
        self.assertEqual(len(queries), len(set(queries)))

    def test_one_edited_document_is_above_the_drift_limit(self):
        # The pipeline's drift limit is a share of the pairs. This ties that
        # share to the shape of the shipped file: every document answers the
        # same number of pairs, and that number is above the limit, so one
        # edited document fails the data stage instead of warning. The README
        # and expert.yaml state it that way.
        per_document = collections.Counter(p["positive"] for p in self.pairs)
        self.assertEqual(len(set(per_document.values())), 1,
                         "the documents no longer answer the same number of pairs")
        share = max(per_document.values()) / len(self.pairs)
        self.assertGreater(share, main.COMMITTED_PAIR_DRIFT_LIMIT)


if __name__ == "__main__":
    unittest.main()
