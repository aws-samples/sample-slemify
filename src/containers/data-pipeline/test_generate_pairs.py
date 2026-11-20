"""main._generate_pairs picks its chunks with a fixed seed and returns the
pairs in selection order, so the same corpus is mined in the same order in
every account, and the train/eval split falls between chunks. The questions
written for those chunks come from a model call per chunk, so how many pairs
each chunk yields still varies and is not part of this guarantee. Run from
this directory: python3 -m unittest test_generate_pairs -v
"""
import os
import random
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

import main  # noqa: E402

# 300 documents, each long enough to become one chunk.
CORPUS = [{"source": f"doc-{i}.md", "content": f"Paragraph {i}. " + ("filler words " * 60)}
          for i in range(300)]


# The 30 chunks the seeded sample picks from CORPUS, in selection order. The
# pairs have to come back in this order whichever thread finishes first.
SELECTION_ORDER = [
    "doc-246.md", "doc-264.md", "doc-110.md", "doc-10.md", "doc-130.md",
    "doc-289.md", "doc-227.md", "doc-185.md", "doc-28.md", "doc-140.md",
    "doc-25.md", "doc-93.md", "doc-144.md", "doc-14.md", "doc-249.md",
    "doc-259.md", "doc-169.md", "doc-153.md", "doc-287.md", "doc-282.md",
    "doc-29.md", "doc-3.md", "doc-118.md", "doc-115.md", "doc-244.md",
    "doc-31.md", "doc-85.md", "doc-161.md", "doc-198.md", "doc-1.md",
]
# The text of the chunk selected first. Its document number is the one in
# SELECTION_ORDER[0].
FIRST_SELECTED_PREFIX = "Paragraph 246."


def generate(target_pairs=30, slow_first=False):
    """Run _generate_pairs with the model call stubbed out. With slow_first,
    only the chunk selected first is delayed, so every other chunk finishes
    before it and completion order cannot match selection order."""
    def queries(backend, domain, chunk_text, n_queries):
        if slow_first and chunk_text.startswith(FIRST_SELECTED_PREFIX):
            time.sleep(0.5)
        return [f"question about {chunk_text[:16]}"]

    with mock.patch.object(main, "_select_backend", lambda model, endpoint: object()), \
            mock.patch.object(main, "_calculate_concurrency", lambda model: 8), \
            mock.patch.object(main, "_generate_queries_for_chunk", queries):
        return main._generate_pairs(raw_content=CORPUS, model="m", endpoint="",
                                    target_pairs=target_pairs, domain="d", queries_per_chunk=1)


class ChunkSampleTest(unittest.TestCase):
    def test_two_runs_pick_the_same_chunks_in_the_same_order(self):
        first, _ = generate()
        # A different global random state is what a second account has.
        random.seed(7)
        second, _ = generate()
        self.assertEqual([p["positive"] for p in first], [p["positive"] for p in second])
        self.assertEqual([p["source"] for p in first], [p["source"] for p in second])

    def test_the_global_random_state_is_left_alone(self):
        random.seed(11)
        expected = random.random()
        random.seed(11)
        generate()
        self.assertEqual(random.random(), expected)

    def test_the_sample_order_is_the_one_this_test_expects(self):
        pairs, _ = generate()
        self.assertEqual([p["source"] for p in pairs], SELECTION_ORDER)

    def test_pair_order_follows_the_chunks_not_the_threads(self):
        # The chunk selected first returns last here, so collecting results as
        # they complete would put its pair at the end instead of the front.
        ordered, _ = generate(slow_first=True)
        self.assertEqual([p["source"] for p in ordered], SELECTION_ORDER)
        self.assertEqual(ordered[0]["source"], SELECTION_ORDER[0])

    def test_the_corpus_comes_back_whole(self):
        pairs, chunks = generate()
        self.assertEqual(len(chunks), len(CORPUS))
        self.assertEqual(len(pairs), 30)


class ChunkSplitTest(unittest.TestCase):
    """The train/eval split falls between chunks, so the queries written for
    one chunk do not end up on both sides of it."""

    # Three chunks with three, two and four queries each: the counts a model
    # call returns in practice.
    PAIRS = ([{"positive": "A"}] * 3 + [{"positive": "B"}] * 2 + [{"positive": "C"}] * 4)

    def split(self, pairs, ratio):
        return main._chunk_split_index(pairs, int(len(pairs) * ratio))

    def test_a_split_inside_a_chunk_moves_to_its_start(self):
        # 90% of 9 pairs is index 8, the third query of chunk C.
        idx = self.split(self.PAIRS, 0.9)
        self.assertEqual(idx, 5)
        train = [p["positive"] for p in self.PAIRS[:idx]]
        held_out = [p["positive"] for p in self.PAIRS[idx:]]
        self.assertEqual(train, ["A", "A", "A", "B", "B"])
        self.assertEqual(held_out, ["C", "C", "C", "C"])
        self.assertFalse(set(train) & set(held_out))

    def test_a_split_already_on_a_boundary_does_not_move(self):
        self.assertEqual(main._chunk_split_index(self.PAIRS, 3), 3)
        self.assertEqual(main._chunk_split_index(self.PAIRS, 5), 5)

    def test_the_ends_and_a_single_chunk_are_left_alone(self):
        self.assertEqual(main._chunk_split_index(self.PAIRS, 0), 0)
        self.assertEqual(main._chunk_split_index(self.PAIRS, 9), 9)
        one_chunk = [{"positive": "A"}] * 4
        self.assertEqual(main._chunk_split_index(one_chunk, 3), 3)


if __name__ == "__main__":
    unittest.main()
