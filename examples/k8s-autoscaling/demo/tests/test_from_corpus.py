"""scripts/index-knowledge.py: --from-corpus loads a JSONL snapshot and
a failed blog fetch is a warning."""
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests import _path

# gitpython shells out to git at import, which is not needed here. Stub it.
sys.modules.setdefault("git", types.SimpleNamespace(Repo=object))
index_knowledge = _path.load_script("scripts/index-knowledge.py", "index_knowledge")

try:
    import bs4  # noqa: F401
except ImportError:
    # Same stand-in as test_index_superseded: find("main").get_text() returns
    # the raw text. Here every fetch fails before it is used.
    class _Soup:
        def __init__(self, text, _parser):
            self.text = text
            self.body = self

        def find_all(self, _names):
            return []

        def find(self, _name):
            return self

        def get_text(self, separator="\n", strip=True):
            return self.text

    sys.modules["bs4"] = types.SimpleNamespace(BeautifulSoup=_Soup)


def rows(n):
    return [{"text": f"chunk {i} text", "source": "karpenter", "section": f"s{i}"} for i in range(n)]


class FromCorpusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "corpus.jsonl")

    def write(self, text):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def run_main(self):
        out = io.StringIO()
        client = mock.Mock()
        argv = ["index-knowledge.py", f"--from-corpus={self.path}"]
        with mock.patch.object(index_knowledge.sys, "argv", argv), \
                mock.patch.object(index_knowledge.sys, "stdout", out), \
                mock.patch.object(index_knowledge, "INDEX_NAME", "test-index"), \
                mock.patch.object(index_knowledge, "get_opensearch_client", return_value=client) as get_client, \
                mock.patch.object(index_knowledge, "create_index"), \
                mock.patch.object(index_knowledge, "index_chunks", return_value=3) as index_chunks, \
                mock.patch.object(index_knowledge, "verify_index_count"), \
                mock.patch.object(index_knowledge, "clone_and_chunk") as clone, \
                mock.patch.object(index_knowledge, "fetch_blogs") as blogs, \
                mock.patch.object(index_knowledge, "chunk_local_docs") as curated:
            try:
                index_knowledge.main()
            finally:
                self.out = out.getvalue()
                self.get_client = get_client
                self.index_chunks = index_chunks
                self.skipped = (clone, blogs, curated)

    def test_valid_file_loads_chunks(self):
        self.write("\n".join(json.dumps(r) for r in rows(3)) + "\n\n")
        self.run_main()
        chunks = self.index_chunks.call_args.args[1]
        self.assertEqual(chunks, rows(3))
        for skipped in self.skipped:
            skipped.assert_not_called()

    def test_bad_line_exits_with_line_number(self):
        lines = [json.dumps(r) for r in rows(2)] + [json.dumps({"text": "", "source": "x", "section": "y"})]
        self.write("\n".join(lines))
        with self.assertRaises(SystemExit) as cm:
            self.run_main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn(f"{self.path}:3:", self.out)
        self.get_client.assert_not_called()

    def test_empty_file_exits(self):
        self.write("\n\n")
        with self.assertRaises(SystemExit) as cm:
            self.run_main()
        self.assertEqual(cm.exception.code, 1)
        self.get_client.assert_not_called()


class FetchBlogsFailureTest(unittest.TestCase):
    def test_failed_fetch_warns_and_continues(self):
        dead = "https://aws.amazon.com/blogs/containers/announcing-karpenter-1-0/"
        live = "https://aws.amazon.com/blogs/containers/cost-optimization-for-kubernetes-on-aws/"

        class Resp:
            text = "<html><body><main>" + ("Karpenter text. " * 40) + "</main></body></html>"

            def raise_for_status(self):
                pass

        def get(url, **kw):
            if url == dead:
                raise OSError("404 Client Error")
            return Resp()
        out = io.StringIO()
        with mock.patch.object(index_knowledge, "BLOG_URLS", [dead, live]), \
                mock.patch.object(index_knowledge, "SUPERSEDED_SOURCES", []), \
                mock.patch.object(index_knowledge.sys, "stdout", out), \
                mock.patch("requests.get", side_effect=get):
            chunks = index_knowledge.fetch_blogs()
        self.assertTrue(chunks)
        self.assertIn("WARNING: 1 blog posts could not be fetched", out.getvalue())
        self.assertIn(f"    {dead}\n", out.getvalue())

    def test_strict_empty_corpus_still_fails(self):
        # The central prep runs --export-only --strict. Dead links are only
        # warnings, but a corpus with 0 chunks still fails the run.
        argv = ["index-knowledge.py", "--export-only", "--strict"]
        with mock.patch.object(index_knowledge.sys, "argv", argv), \
                mock.patch.object(index_knowledge.sys, "stdout", io.StringIO()) as out, \
                mock.patch.object(index_knowledge, "clone_and_chunk", return_value=[]), \
                mock.patch.object(index_knowledge, "chunk_local_docs", return_value=[]), \
                mock.patch.object(index_knowledge, "BLOG_URLS", ["https://x/blogs/a/"]), \
                mock.patch.object(index_knowledge, "SUPERSEDED_SOURCES", []), \
                mock.patch("requests.get", side_effect=OSError("404 Client Error")):
            with self.assertRaises(SystemExit) as cm:
                index_knowledge.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Error: 0 chunks (--strict)", out.getvalue())


if __name__ == "__main__":
    unittest.main()
