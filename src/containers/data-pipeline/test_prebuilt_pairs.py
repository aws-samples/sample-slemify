"""A retriever project can take its training pairs from a prebuilt file in the
bucket. The pipeline uses the file when every check passes and generates the
pairs when any check fails. Run from this directory:
python3 -m unittest test_prebuilt_pairs -v
"""
import hashlib
import json
import os
import sys
import unittest
from unittest import mock
from botocore.exceptions import ClientError
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
import main  # noqa: E402
PATH = "k8s-autoscaling/data/"
PROJECT = "k8s-autoscaling-retriever"
BUCKET = "team-models"
MANIFEST_KEY = PATH + "prebuilt/" + PROJECT + "/manifest.json"
TRAIN_KEY = PATH + "prebuilt/" + PROJECT + "/train.jsonl"
SOURCES = [{"source": f"{PATH}queries/doc-{i}.txt",
            "content": f"Document {i}. " + ("filler words " * 40)} for i in range(12)]
TARGET = 30
def sha(data):
    return hashlib.sha256(data).hexdigest()
def error(code):
    return ClientError({"Error": {"Code": code, "Message": "stub"}}, "GetObject")
class FakeS3:
    """Objects in a dict; a key can be set to raise an error code instead."""
    def __init__(self, objects=None, errors=None):
        self.objects = dict(objects or {})
        self.errors = dict(errors or {})
        self.puts = {}
        self.deletes = []
    def get_object(self, Bucket, Key):
        if Key in self.errors:
            raise error(self.errors[Key])
        if Key not in self.objects:
            raise error("NoSuchKey")
        body = self.objects[Key]
        return {"Body": mock.Mock(read=lambda: body)}
    def put_object(self, Bucket, Key, Body):
        self.puts[Key] = Body
    def delete_object(self, Bucket, Key):
        self.deletes.append(Key)
def artifact(chunks=None, per_chunk=3, **manifest_overrides):
    """A valid train.jsonl and manifest for the chunks of SOURCES."""
    chunks = chunks if chunks is not None else main._chunk_documents(SOURCES)
    pairs = [{"query": f"question {q} about {c['text'][:12]}", "positive": c["text"], "source": c["source"]}
             for c in chunks[:10] for q in range(per_chunk)]
    body = "\n".join(json.dumps(p) for p in pairs).encode("utf-8")
    manifest = {"version": 1, "project": PROJECT, "tag": "v0.3.0", "commit": "a" * 40,
                "model": "m", "pairs": len(pairs), "chunks": len(chunks), "expected": 30,
                "bytes": len(body), "sha256": sha(body), "corpus_sha256": main._corpus_sha256(chunks),
                "generatedAt": "2026-11-24T10:00:00Z"}
    manifest.update(manifest_overrides)
    return body, manifest
def objects(body, manifest):
    return {TRAIN_KEY: body, MANIFEST_KEY: json.dumps(manifest).encode("utf-8")}
class ReadPrebuiltTest(unittest.TestCase):
    def read(self, s3, chunks=None):
        chunks = chunks if chunks is not None else main._chunk_documents(SOURCES)
        with mock.patch.object(main.boto3, "client", lambda *a, **k: s3):
            return main._read_prebuilt_pairs(BUCKET, PATH, PROJECT, chunks,
                                             main._expected_pairs(len(chunks), TARGET), TARGET)
    def assert_rejected(self, s3, reason, level="WARNING", chunks=None):
        with self.assertLogs(main.logger, level="INFO") as logs:
            self.assertIsNone(self.read(s3, chunks))
        lines = [m for m in logs.output if "not used" in m or "No prebuilt pairs" in m]
        self.assertEqual(len(lines), 1, logs.output)
        self.assertIn(level, lines[0])
        self.assertIn(reason, lines[0])
        self.assertEqual([m for m in logs.output if "Using" in m], [])
    def test_a_valid_artifact_is_used(self):
        body, manifest = artifact()
        with self.assertLogs(main.logger, level="INFO") as logs:
            pairs = self.read(FakeS3(objects(body, manifest)))
        self.assertEqual(len(pairs), 30)
        self.assertEqual(pairs[0]["source"], f"{PATH}queries/doc-0.txt")
        [line] = [m for m in logs.output if "Using" in m]
        self.assertIn("30 prebuilt pairs", line)
        self.assertIn("aaaaaaaaaaaa", line)
        self.assertIn("2026-11-24T10:00:00Z", line)
        self.assertIn("no Bedrock calls", line)
    def test_the_target_caps_the_pairs(self):
        body, manifest = artifact()
        chunks = main._chunk_documents(SOURCES)
        with mock.patch.object(main.boto3, "client", lambda *a, **k: FakeS3(objects(body, manifest))):
            pairs = main._read_prebuilt_pairs(BUCKET, PATH, PROJECT, chunks, 30, 20)
        self.assertEqual(len(pairs), 20)
    def test_no_manifest_is_information_not_a_warning(self):
        self.assert_rejected(FakeS3(), PATH + "prebuilt/", level="INFO")
    def test_an_s3_error_is_a_warning_with_its_code(self):
        body, manifest = artifact()
        s3 = FakeS3(objects(body, manifest), errors={MANIFEST_KEY: "AccessDenied"})
        self.assert_rejected(s3, "AccessDenied")
    def test_a_failure_to_read_train_jsonl_is_a_warning(self):
        body, manifest = artifact()
        s3 = FakeS3(objects(body, manifest), errors={TRAIN_KEY: "InternalError"})
        self.assert_rejected(s3, "reading train.jsonl failed: InternalError")
    def test_manifest_checks(self):
        body, manifest = artifact()
        good = objects(body, manifest)
        cases = [
            ("not JSON", {MANIFEST_KEY: b"{oops"}),
            ("version is not 1", {MANIFEST_KEY: json.dumps(dict(manifest, version=2)).encode()}),
            ("another project", {MANIFEST_KEY: json.dumps(dict(manifest, project="other")).encode()}),
            ("pairs is not a positive integer", {MANIFEST_KEY: json.dumps(dict(manifest, pairs=0)).encode()}),
            ("bytes is not a positive integer", {MANIFEST_KEY: json.dumps(dict(manifest, bytes="5")).encode()}),
            ("chunks is not a positive integer", {MANIFEST_KEY: json.dumps(dict(manifest, chunks=True)).encode()}),
            ("sha256 is not 64 lowercase hex", {MANIFEST_KEY: json.dumps(dict(manifest, sha256="ABC")).encode()}),
            ("corpus_sha256 is not 64 lowercase hex",
             {MANIFEST_KEY: json.dumps(dict(manifest, corpus_sha256=body.hex()[:63])).encode()}),
            ("version is not 1", {MANIFEST_KEY: b"[1]"}),
        ]
        for reason, override in cases:
            with self.subTest(reason=reason):
                self.assert_rejected(FakeS3(dict(good, **override)), reason)
    def test_train_file_checks(self):
        body, manifest = artifact()
        chunks = main._chunk_documents(SOURCES)
        def with_body(new_body, **over):
            m = dict(manifest, bytes=len(new_body), sha256=sha(new_body), **over)
            return FakeS3(objects(new_body, m))
        lines = body.decode().splitlines()
        cases = [
            ("size differs", FakeS3(objects(body + b"\n", manifest))),
            ("sha256 differs", FakeS3(objects(body.replace(b"question 0", b"question X", 1), manifest))),
            ("other source documents", FakeS3(objects(body, dict(manifest, corpus_sha256="0" * 64)))),
            ("not a (query, positive) pair", with_body("\n".join(
                lines[:-1] + [json.dumps({"query": "", "positive": lines[0]})]).encode(), pairs=30)),
            ("not a (query, positive) pair", with_body("\n".join(lines[:-1] + ["[1]"]).encode(), pairs=30)),
            ("not JSON", with_body("\n".join(lines[:-1] + ["{broken"]).encode(), pairs=30)),
            ("not in the sources", with_body("\n".join(lines[:-1] + [json.dumps(
                {"query": "q", "positive": "a document no source holds"})]).encode(), pairs=30)),
            ("line count differs", with_body("\n".join(lines[:-1]).encode(), pairs=30)),
            ("is under 50%", with_body("\n".join(lines[:14]).encode(), pairs=14)),
        ]
        for reason, s3 in cases:
            with self.subTest(reason=reason):
                self.assert_rejected(s3, reason, chunks=chunks)
    def test_the_floor_itself_is_accepted(self):
        body, manifest = artifact()
        lines = body.decode().splitlines()[:15]
        new_body = "\n".join(lines).encode()
        manifest.update(pairs=15, bytes=len(new_body), sha256=sha(new_body))
        self.assertEqual(len(self.read(FakeS3(objects(new_body, manifest)))), 15)
    def test_pairs_for_edited_documents_are_rejected(self):
        # A source file changed after the pairs were written.
        body, manifest = artifact()
        edited = [dict(s) for s in SOURCES]
        edited[0]["content"] += " An extra sentence that changes the chunk."
        self.assert_rejected(FakeS3(objects(body, manifest)), "other source documents",
                             chunks=main._chunk_documents(edited))
class PipelineUsesPrebuiltTest(unittest.TestCase):
    DATA_CFG = {"bucket": BUCKET, "path": PATH, "sources": [{"path": "queries/"}]}
    def run_pipeline(self, s3, generate=None):
        written = {}
        generate = generate or mock.Mock(side_effect=AssertionError("generated"))
        with mock.patch.object(main.boto3, "client", lambda *a, **k: s3), \
                mock.patch.object(main, "_generate_pairs_stats", generate), \
                mock.patch.object(main, "write_jsonl_to_s3",
                                  lambda bucket, key, records: written.__setitem__(key, records)):
            main.run_embedding_pipeline({}, self.DATA_CFG, PROJECT, "d", SOURCES,
                                        {"model": "m", "pairs": TARGET})
        return written, generate
    def test_prebuilt_pairs_become_the_training_pairs_and_nothing_is_generated(self):
        body, manifest = artifact()
        written, generate = self.run_pipeline(FakeS3(objects(body, manifest)))
        generate.assert_not_called()
        wanted = [json.loads(l) for l in body.decode().splitlines()]
        train, held_out = written[PROJECT + "/processed/train.jsonl"], written[PROJECT + "/processed/eval.jsonl"]
        # No committed held-out pairs here, so the split of the pairs is used.
        self.assertEqual([p["query"] for p in train + held_out], [p["query"] for p in wanted])
        self.assertEqual(len(written[PROJECT + "/processed/corpus.jsonl"]), 12)
    def test_an_invalid_artifact_falls_back_to_generation(self):
        body, manifest = artifact()
        manifest["sha256"] = "0" * 64
        chunks = main._chunk_documents(SOURCES)
        pairs = [{"query": "q%d" % i, "positive": chunks[i % 10]["text"], "source": "s"} for i in range(30)]
        stats = {"expected": 30, "chunks": 12, "rounds": 0, "seconds": 0.0, "errors": {}}
        generate = mock.Mock(return_value=(pairs, chunks, stats))
        with self.assertLogs(main.logger, level="INFO") as logs:
            written, generate = self.run_pipeline(FakeS3(objects(body, manifest)), generate)
        generate.assert_called_once()
        self.assertTrue(any("sha256 differs" in m for m in logs.output))
        self.assertEqual(len(written[PROJECT + "/processed/train.jsonl"])
                         + len(written[PROJECT + "/processed/eval.jsonl"]), 30)
    def test_no_artifact_generates_as_before(self):
        chunks = main._chunk_documents(SOURCES)
        pairs = [{"query": "q%d" % i, "positive": chunks[i % 10]["text"], "source": "s"} for i in range(30)]
        stats = {"expected": 30, "chunks": 12, "rounds": 0, "seconds": 0.0, "errors": {}}
        generate = mock.Mock(return_value=(pairs, chunks, stats))
        with self.assertLogs(main.logger, level="INFO") as logs:
            _, generate = self.run_pipeline(FakeS3(), generate)
        generate.assert_called_once()
        self.assertEqual([m for m in logs.output if "WARNING" in m], [])
        self.assertTrue(any("No prebuilt pairs" in m for m in logs.output))
if __name__ == "__main__":
    unittest.main()
