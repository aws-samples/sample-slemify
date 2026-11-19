"""main._generate_pairs picks its chunks with a fixed seed and returns the
pairs in selection order, so the same corpus gives the same documents and the
same train/eval split in every account. The questions written for those chunks
come from a model call and are not part of this guarantee. Run from this
directory: python3 -m unittest test_generate_pairs -v
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


def generate(target_pairs=30, slow_first=False):
    """Run _generate_pairs with the model call stubbed out. With slow_first,
    the first chunk submitted finishes last, so completion order differs from
    selection order."""
    def queries(backend, domain, chunk_text, n_queries):
        if slow_first and chunk_text.startswith("Paragraph") and " 1." not in chunk_text[:14]:
            time.sleep(0.02)
        return [f"question about {chunk_text[:12]}"]

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

    def test_pair_order_follows_the_chunks_not_the_threads(self):
        ordered, _ = generate(slow_first=True)
        plain, _ = generate()
        self.assertEqual([p["source"] for p in ordered], [p["source"] for p in plain])

    def test_the_corpus_comes_back_whole(self):
        pairs, chunks = generate()
        self.assertEqual(len(chunks), len(CORPUS))
        self.assertEqual(len(pairs), 30)


if __name__ == "__main__":
    unittest.main()
