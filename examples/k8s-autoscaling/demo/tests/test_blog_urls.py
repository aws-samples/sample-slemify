"""scripts/check-blog-urls.py: BLOG_URLS has no duplicate slugs, and a dead
URL fails the check that the release workflow runs."""
import io
import unittest
import urllib.error
from unittest import mock

from tests import _path

check = _path.load_script("scripts/check-blog-urls.py", "check_blog_urls")


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _opener(dead):
    def opener(req, timeout):
        if req.full_url in dead:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        return _Resp()
    return opener


class BlogUrlTest(unittest.TestCase):
    def test_lists_load_and_slugs_are_unique(self):
        urls, superseded = check.load_lists()
        self.assertTrue(urls)
        self.assertTrue(superseded <= set(urls))
        self.assertEqual(check.duplicate_slugs(urls), [])

    def test_duplicate_slug_is_found(self):
        urls = ["https://a/blogs/containers/x/", "https://a/blogs/mt/x/"]
        self.assertEqual(check.duplicate_slugs(urls), ["x"])

    def test_all_live_passes(self):
        with mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(check.main(opener=_opener(set()), sleep=lambda s: None), 0)

    def test_dead_url_fails(self):
        urls, superseded = check.load_lists()
        dead = next(u for u in urls if u not in superseded)
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            self.assertEqual(check.main(opener=_opener({dead}), sleep=lambda s: None), 1)
        self.assertIn(f"404 {dead}", out.getvalue())

    def test_server_error_is_retried(self):
        calls = []

        def opener(req, timeout):
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", {}, None)
            return _Resp()
        self.assertEqual(check.status("https://x/", opener, lambda s: None), 200)
        self.assertEqual(len(calls), 3)

    def test_not_found_is_not_retried(self):
        calls = []

        def opener(req, timeout):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        self.assertEqual(check.status("https://x/", opener, lambda s: None), 404)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
