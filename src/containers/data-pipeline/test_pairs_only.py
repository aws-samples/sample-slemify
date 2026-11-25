"""--pairs-only writes a retriever project's training pairs and manifest from
the source files on disk, for the central prep build. The records, the chunk
list and the seeded chunk sample have to be the ones a Job builds from the
same files in S3, and the team has to accept what prep writes. Run from this
directory: python3 -m unittest test_pairs_only -v
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
import main  # noqa: E402
from test_prebuilt_pairs import MANIFEST_KEY, PROJECT, TRAIN_KEY  # noqa: E402
from test_pair_rounds import FakeClock, StubBackend  # noqa: E402
REPO_EXAMPLE = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "examples", "k8s-autoscaling"))
PATH = "k8s-autoscaling/data/"
class ListingS3:
    """S3 for read_raw_sources: list_objects_v2 returns keys in UTF-8 byte
    order, as S3 does."""
    def __init__(self, files, extra=None):
        self.files = files
        self.extra = extra or {}
    def get_paginator(self, name):
        files = self.files
        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted((k for k in files if k.startswith(Prefix)), key=lambda k: k.encode("utf-8"))
                yield {"Contents": [{"Key": k} for k in keys]}
        return Paginator()
    def get_object(self, Bucket, Key):
        body = self.extra[Key] if Key in self.extra else self.files[Key]
        return {"Body": mock.Mock(read=lambda: body)}
class LocalSourcesTest(unittest.TestCase):
    FILES = {
        "queries/b-second.txt": b"Second document.\n\n" + b"words " * 60,
        "queries/A-upper.txt": b"  Upper case name sorts first.  \n" + b"more " * 60,
        "queries/a_lower.txt": b"Lower case name. " + b"text " * 60,
        "queries/a-dash.txt": b"Dash sorts before underscore. " + b"data " * 60,
        "queries/\u00e9-accent.txt": "Accent sorts last. ".encode("utf-8") + b"abc " * 60,
        "queries/sub/nested.txt": b"A nested file is uploaded too. " + b"deep " * 60,
        "queries/empty.txt": b"  \n\n",
        "queries/bad-utf8.txt": b"Broken byte \xff inside. " + b"x " * 120,
        "queries/ignored.md": b"Not uploaded: not a txt file.",
    }
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.dir, True)
        for rel, data in self.FILES.items():
            path = os.path.join(self.dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
    def from_s3(self):
        s3_files = {PATH + rel: data for rel, data in self.FILES.items() if rel.endswith(".txt")}
        with mock.patch.object(main.boto3, "client", lambda *a, **k: ListingS3(s3_files)):
            return main.read_raw_sources("b", PATH, [{"path": "queries/", "type": "raw"}])
    def test_local_records_equal_the_records_a_job_reads_from_s3(self):
        local = main._read_local_sources(self.dir, PATH, [{"path": "queries/", "type": "raw"}])
        self.assertEqual(local, self.from_s3())
        self.assertEqual(len(local), 7)
        self.assertTrue(all(r["source"].startswith(PATH + "queries/") for r in local))
    def test_chunks_and_the_seeded_sample_are_the_same(self):
        local = main._read_local_sources(self.dir, PATH, [{"path": "queries/"}])
        a, b = main._chunk_documents(local), main._chunk_documents(self.from_s3())
        self.assertEqual(a, b)
        n = max(1, len(a) // 2)
        import random
        self.assertEqual(random.Random(main.CHUNK_SAMPLE_SEED).sample(a, n),
                         random.Random(main.CHUNK_SAMPLE_SEED).sample(b, n))
    def test_the_shipped_queries_read_the_same_either_way(self):
        queries = os.path.join(REPO_EXAMPLE, "data", "queries")
        files = {}
        for dirpath, _d, names in os.walk(queries):
            for name in names:
                if name.endswith(".txt"):
                    full = os.path.join(dirpath, name)
                    with open(full, "rb") as f:
                        files[PATH + os.path.relpath(full, os.path.join(REPO_EXAMPLE, "data"))] = f.read()
        self.assertGreater(len(files), 50)
        with mock.patch.object(main.boto3, "client", lambda *a, **k: ListingS3(files)):
            from_s3 = main.read_raw_sources("b", PATH, [{"path": "queries/"}])
        local = main._read_local_sources(os.path.join(REPO_EXAMPLE, "data"), PATH, [{"path": "queries/"}])
        self.assertEqual(local, from_s3)
        self.assertEqual(main._chunk_documents(local), main._chunk_documents(from_s3))
def write_config(directory, name="expert.yaml", **synthetic):
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write("""project: {name: %s, task: embedding, domain: Kubernetes autoscaling}
data:
  bucket: slemify-data
  path: %s
  sources:
    - {path: queries/, type: raw}
  synthetic: {pairs: 30%s}
""" % (PROJECT, PATH, "".join(", %s: %s" % kv for kv in synthetic.items())))
    return path
class PairsOnlyTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.dir, True)
        self.sources = os.path.join(self.dir, "data")
        os.makedirs(os.path.join(self.sources, "queries"))
        for i in range(12):
            with open(os.path.join(self.sources, "queries", f"doc-{i:02d}.txt"), "w") as f:
                f.write(f"Document {i}. " + "filler words " * 40)
        self.out = os.path.join(self.dir, "train.jsonl")
        self.manifest = os.path.join(self.dir, "manifest.json")
        clock = FakeClock()
        for p in (mock.patch.object(main, "_calculate_concurrency", lambda m: 8),
                  mock.patch.object(main, "_monotonic", clock.monotonic),
                  mock.patch.object(main, "_pause", clock.pause),
                  mock.patch.object(main._JITTER, "uniform", lambda a, b: 0),
                  mock.patch.object(main.time, "sleep", lambda s: None)):
            p.start()
            self.addCleanup(p.stop)
    def run_only(self, backend, config=None, env=None, extra=None):
        argv = ["--pairs-only", "--config", config or write_config(self.dir, model="the-model"),
                "--sources-dir", self.sources, "--out", self.out, "--manifest", self.manifest] + (extra or [])
        env = dict({"RELEASE_TAG": "v0.3.0", "CODEBUILD_RESOLVED_SOURCE_VERSION": "b" * 40}, **(env or {}))
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(main, "_select_backend", lambda model, endpoint: backend):
            return main.pairs_only(argv)
    def three(self):
        return StubBackend(reply="\n".join(json.dumps({"query": "q%d" % i}) for i in range(3)))
    def test_writes_the_pairs_and_a_manifest_that_describes_them(self):
        self.assertEqual(self.run_only(self.three()), 0)
        with open(self.out, "rb") as f:
            body = f.read()
        with open(self.manifest) as f:
            m = json.load(f)
        pairs = [json.loads(l) for l in body.decode().splitlines()]
        self.assertEqual(len(pairs), 30)
        self.assertEqual(set(pairs[0]), {"query", "positive", "source"})
        self.assertEqual(m["version"], 1)
        self.assertEqual(m["project"], PROJECT)
        self.assertEqual((m["tag"], m["commit"]), ("v0.3.0", "b" * 40))
        self.assertEqual((m["model"], m["pairs"], m["chunks"], m["expected"]), ("the-model", 30, 12, 30))
        self.assertEqual((m["bytes"], m["sha256"]), (len(body), hashlib.sha256(body).hexdigest()))
        self.assertRegex(m["generatedAt"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertRegex(m["corpus_sha256"], r"^[0-9a-f]{64}$")
    def test_a_team_accepts_what_prep_wrote(self):
        self.assertEqual(self.run_only(self.three()), 0)
        files = {}
        for rel in os.listdir(os.path.join(self.sources, "queries")):
            with open(os.path.join(self.sources, "queries", rel), "rb") as f:
                files[PATH + "queries/" + rel] = f.read()
        with open(self.out, "rb") as f:
            train = f.read()
        with open(self.manifest, "rb") as f:
            manifest = f.read()
        s3 = ListingS3(files, extra={TRAIN_KEY: train, MANIFEST_KEY: manifest})
        with mock.patch.object(main.boto3, "client", lambda *a, **k: s3):
            raw = main.read_raw_sources("b", PATH, [{"path": "queries/"}])
            chunks = main._chunk_documents(raw)
            pairs = main._read_prebuilt_pairs("b", PATH, PROJECT, chunks,
                                              main._expected_pairs(len(chunks), 30), 30)
        self.assertEqual(len(pairs), 30)
    def test_the_model_comes_from_the_config_then_the_environment_then_the_default(self):
        seen = []
        def fake(raw_content, model, endpoint, target_pairs, domain, queries_per_chunk=3):
            seen.append(model)
            return [], [], {"expected": 0, "chunks": 0, "rounds": 0, "seconds": 0.0, "errors": {}}
        plain = write_config(self.dir, name="plain.yaml")
        with mock.patch.object(main, "_generate_pairs_stats", fake):
            self.run_only(None, config=write_config(self.dir, model="from-config"),
                          env={"SLEMIFY_BEDROCK_MODEL": "from-env"})
            self.run_only(None, config=plain, env={"SLEMIFY_BEDROCK_MODEL": "from-env"})
            with mock.patch.dict(os.environ):
                os.environ.pop("SLEMIFY_BEDROCK_MODEL", None)
                self.run_only(None, config=plain)
        self.assertEqual(seen, ["from-config", "from-env", "global.anthropic.claude-sonnet-5"])
    def test_an_outage_that_never_ends_fails_and_writes_nothing(self):
        self.assertEqual(self.run_only(StubBackend(fail_calls=None)), 1)
        self.assertFalse(os.path.exists(self.out))
        self.assertFalse(os.path.exists(self.manifest))
    def test_a_short_outage_is_ridden_out(self):
        # The first pass fails every call (10 chunks, 3 attempts each).
        backend = StubBackend(fail_calls=30, reply=json.dumps({"query": "q"}))
        # One query per chunk is 10 of 30 expected: under the floor.
        self.assertEqual(self.run_only(backend), 1)
        backend = StubBackend(fail_calls=30, reply="\n".join(json.dumps({"query": "q%d" % i}) for i in range(3)))
        self.assertEqual(self.run_only(backend), 0)
        with open(self.manifest) as f:
            self.assertEqual(json.load(f)["pairs"], 30)
    def test_other_tasks_and_empty_sources_fail(self):
        config = write_config(self.dir)
        with open(config) as f:
            text = f.read()
        with open(config, "w") as f:
            f.write(text.replace("task: embedding", "task: classification"))
        self.assertEqual(self.run_only(self.three(), config=config), 1)
        for name in os.listdir(os.path.join(self.sources, "queries")):
            os.remove(os.path.join(self.sources, "queries", name))
        self.assertEqual(self.run_only(self.three()), 1)
    def test_main_dispatches_to_pairs_only(self):
        with mock.patch.object(sys, "argv", ["main.py", "--pairs-only", "--config", "x"]), \
                mock.patch.object(main, "pairs_only", lambda argv: 7):
            with self.assertRaises(SystemExit) as cm:
                main.main()
        self.assertEqual(cm.exception.code, 7)
if __name__ == "__main__":
    unittest.main()
