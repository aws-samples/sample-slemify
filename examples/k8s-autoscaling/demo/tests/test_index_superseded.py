"""scripts/index-knowledge.py: SUPERSEDED_SOURCES is well formed and
fetch_blogs() never fetches or chunks a superseded post."""
import io
import sys
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
    # Minimal stand-in: BeautifulSoup(html).find("main").get_text() returns the
    # raw text, which is all fetch_blogs() needs here.
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

SUPERSEDED = index_knowledge.SUPERSEDED_SOURCES[0]["source"]
KEPT = "https://aws.amazon.com/blogs/containers/announcing-karpenter-1-0/"
PAGE = "<html><body><main>" + ("Karpenter v1 text. " * 40) + "</main></body></html>"


def fake_get(url, **_kw):
    resp = mock.Mock()
    resp.text = PAGE
    resp.raise_for_status = lambda: None
    return resp


class SupersededListTest(unittest.TestCase):
    def test_entries_are_complete_and_listed(self):
        self.assertTrue(index_knowledge.SUPERSEDED_SOURCES)
        for entry in index_knowledge.SUPERSEDED_SOURCES:
            for key in ("source", "reason", "replacement"):
                self.assertTrue(entry.get(key, "").strip(), f"{key} empty in {entry}")
            self.assertIn(entry["source"], index_knowledge.BLOG_URLS)

    def test_no_stale_entries(self):
        self.assertEqual(index_knowledge.stale_superseded(), [])


class FetchBlogsTest(unittest.TestCase):
    def run_fetch(self, urls, superseded=None):
        out = io.StringIO()
        patches = [mock.patch.object(index_knowledge, "BLOG_URLS", urls),
                   mock.patch.object(index_knowledge.sys, "stdout", out),
                   mock.patch("requests.get", side_effect=fake_get)]
        if superseded is not None:
            patches.append(mock.patch.object(index_knowledge, "SUPERSEDED_SOURCES", superseded))
        for p in patches:
            p.start()
        try:
            get = sys.modules["requests"].get
            chunks = index_knowledge.fetch_blogs()
            requested = [c.args[0] for c in get.call_args_list]
        finally:
            for p in reversed(patches):
                p.stop()
        return chunks, requested, out.getvalue()

    def test_superseded_url_is_not_requested(self):
        chunks, requested, out = self.run_fetch([SUPERSEDED, KEPT])
        self.assertEqual(requested, [KEPT])
        self.assertTrue(chunks)
        self.assertEqual({c["section"] for c in chunks}, {"announcing-karpenter-1-0"})
        self.assertIn("using-amazon-ec2-spot-instances-with-karpenter: skipped, superseded (", out)
        self.assertIn("1 superseded posts skipped (SUPERSEDED_SOURCES)", out)

    def test_stale_entry_warns_and_does_not_raise(self):
        stale = {"source": "https://example.com/gone/", "reason": "r", "replacement": "x"}
        chunks, requested, out = self.run_fetch([KEPT], superseded=[stale])
        self.assertEqual(requested, [KEPT])
        self.assertTrue(chunks)
        self.assertIn("Warning: SUPERSEDED_SOURCES entry not in BLOG_URLS: https://example.com/gone/", out)


if __name__ == "__main__":
    unittest.main()
