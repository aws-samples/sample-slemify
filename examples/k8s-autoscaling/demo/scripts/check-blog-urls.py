"""Check that every blog post the knowledge corpus fetches still loads.

A dead URL in BLOG_URLS does not fail the corpus export (index-knowledge.py
skips it with a warning), but its post is then missing from the corpus. The
release workflow runs this with --warn-only so a dead link shows up in the
release log without blocking the release.

  python3 check-blog-urls.py            exit 1 if a URL does not return 200
                                        or two URLs share a slug
  python3 check-blog-urls.py --warn-only
                                        print the same report, always exit 0

Reads BLOG_URLS and SUPERSEDED_SOURCES from index-knowledge.py without
importing it, so it needs only the standard library. Superseded posts are not
fetched by the export, so they are not checked.
"""
import ast
import os
import sys
import time
import urllib.error
import urllib.request

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index-knowledge.py")
ATTEMPTS = 3
PAUSE_S = 5


def load_lists(path=SCRIPT):
    """Return (BLOG_URLS, superseded sources) from index-knowledge.py."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("BLOG_URLS", "SUPERSEDED_SOURCES"):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    urls = values["BLOG_URLS"]
    superseded = {e["source"] for e in values.get("SUPERSEDED_SOURCES", [])}
    return urls, superseded


def slug(url):
    return url.rstrip("/").split("/")[-1]


def duplicate_slugs(urls):
    """Slugs that appear more than once. fetch_blogs() names chunks by slug."""
    seen, dups = set(), []
    for u in urls:
        s = slug(u)
        if s in seen and s not in dups:
            dups.append(s)
        seen.add(s)
    return dups


def status(url, opener=urllib.request.urlopen, sleep=time.sleep):
    """HTTP status of a GET, retried on network errors and 5xx."""
    last = None
    for attempt in range(ATTEMPTS):
        req = urllib.request.Request(url, headers={"User-Agent": "slemify/1.0"})
        try:
            with opener(req, timeout=15) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            last = e.code
            if e.code < 500:
                return e.code
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"error: {e}"
        if attempt < ATTEMPTS - 1:
            sleep(PAUSE_S)
    return last


def main(opener=urllib.request.urlopen, sleep=time.sleep, warn_only=False):
    urls, superseded = load_lists()
    bad = []
    for s in duplicate_slugs(urls):
        bad.append(f"duplicate slug {s}")
    for url in urls:
        if url in superseded:
            continue
        code = status(url, opener, sleep)
        print(f"{code} {url}")
        if code != 200:
            bad.append(f"{code} {url}")
    if bad:
        label = "WARNING: blog URL check found problems" if warn_only else "Blog URL check failed"
        print(f"{label}. Remove or replace these entries in BLOG_URLS:")
        for b in bad:
            print(f"  {b}")
        return 0 if warn_only else 1
    print(f"all {len(urls) - len(superseded & set(urls))} fetched blog URLs return 200")
    return 0


if __name__ == "__main__":
    sys.exit(main(warn_only="--warn-only" in sys.argv[1:]))
