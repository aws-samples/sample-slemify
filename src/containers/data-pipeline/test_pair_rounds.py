"""Seed tolerance to a Bedrock outage: rounds that retry only the chunks
without questions, the floor on the pairs a run may keep, the failure record
and the test-event outage window. The model call is a stub backend, and a
fake clock replaces the pauses. Run from this directory:
python3 -m unittest test_pair_rounds -v
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from botocore.exceptions import (
    ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
import main  # noqa: E402
from test_generate_pairs import CORPUS, SELECTION_ORDER  # noqa: E402
def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": "stub"}}, "Converse")
class StubBackend:
    """Raises `code` for the first `fail_calls` calls (None: every call), and
    for prompts that contain `fail_prompt` until `fail_prompt_calls` calls of
    them have failed. Otherwise answers with one query per chunk."""
    def __init__(self, code="ServiceUnavailableException", fail_calls=0,
                 fail_prompt=None, fail_prompt_calls=0, reply=None):
        self.code, self.fail_calls = code, fail_calls
        self.fail_prompt, self.fail_prompt_calls = fail_prompt, fail_prompt_calls
        self.reply = reply
        self.lock = threading.Lock()
        self.calls = 0
        self.prompts = []
    def generate(self, prompt):
        with self.lock:
            self.calls += 1
            self.prompts.append(prompt)
            n = self.calls
            fail = self.fail_calls is None or n <= self.fail_calls
            if self.fail_prompt and self.fail_prompt in prompt and self.fail_prompt_calls > 0:
                self.fail_prompt_calls -= 1
                fail = True
        if fail:
            raise client_error(self.code)
        if self.reply is not None:
            return self.reply
        return json.dumps({"query": "question %d" % n})
class FakeClock:
    """Pauses advance a clock instead of waiting."""
    def __init__(self):
        self.now = 1000.0
        self.pauses = []
    def monotonic(self):
        return self.now
    def pause(self, seconds):
        self.pauses.append(round(seconds))
        self.now += seconds
class RoundsBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.before = main.ERROR_TALLY.snapshot()
        patches = [
            mock.patch.object(main, "_calculate_concurrency", lambda model: 8),
            mock.patch.object(main, "_monotonic", self.clock.monotonic),
            mock.patch.object(main, "_pause", self.clock.pause),
            mock.patch.object(main._JITTER, "uniform", lambda a, b: 0),
            # The per-call pauses inside _call_with_retry.
            mock.patch.object(main.time, "sleep", lambda s: None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
    def run_generation(self, backend, target_pairs=30, corpus=CORPUS):
        self.backend = backend
        with mock.patch.object(main, "_select_backend", lambda model, endpoint: backend):
            return main._generate_pairs_stats(
                raw_content=corpus, model="m", endpoint="",
                target_pairs=target_pairs, domain="d", queries_per_chunk=1)
class RoundsTest(RoundsBase):
    def test_a_short_outage_is_ridden_out_in_one_round(self):
        # Every call of the first pass fails: 30 chunks, 3 attempts each.
        pairs, chunks, stats = self.run_generation(StubBackend(fail_calls=90))
        self.assertEqual(len(pairs), 30)
        self.assertEqual(stats["rounds"], 1)
        self.assertEqual(self.clock.pauses, [30])
        self.assertEqual(stats["errors"], {"ServiceUnavailableException": 90})
        self.assertEqual(stats["expected"], 30)
        self.assertEqual(len(chunks), len(CORPUS))
    def test_the_pairs_come_back_in_selection_order_after_rounds(self):
        pairs, _, _ = self.run_generation(StubBackend(fail_calls=90))
        self.assertEqual([p["source"] for p in pairs], SELECTION_ORDER)
    def test_pauses_grow_and_stop_at_the_budget(self):
        # The model never answers. 30, 60, 120 and 240 s pauses add up to 450 s
        # of the 480 s budget, and the last round waits only for the rest.
        pairs, _, stats = self.run_generation(StubBackend(fail_calls=None))
        self.assertEqual(pairs, [])
        self.assertEqual(self.clock.pauses, [30, 60, 120, 240, 30])
        self.assertEqual(stats["rounds"], 5)
        self.assertLessEqual(self.clock.now - 1000.0, main.PAIR_RETRY_BUDGET_S)
        self.assertEqual(stats["errors"]["ServiceUnavailableException"], 30 * 3 * 6)
    def test_jitter_is_added_to_the_pause(self):
        with mock.patch.object(main._JITTER, "uniform", lambda a, b: 7):
            self.run_generation(StubBackend(fail_calls=90))
        self.assertEqual(self.clock.pauses, [37])
    def test_only_the_chunks_without_questions_are_asked_again(self):
        first = "Paragraph 246."
        backend = StubBackend(fail_prompt=first, fail_prompt_calls=3)
        pairs, _, stats = self.run_generation(backend)
        self.assertEqual(len(pairs), 30)
        self.assertEqual(stats["rounds"], 1)
        # 29 chunks once, the failed one three times and then once more.
        self.assertEqual(backend.calls, 29 + 3 + 1)
        self.assertEqual(sum(1 for p in backend.prompts if first in p), 4)
        self.assertEqual(pairs[0]["source"], SELECTION_ORDER[0])
    def test_a_wrong_model_id_fails_without_rounds(self):
        pairs, _, stats = self.run_generation(StubBackend(code="ValidationException", fail_calls=None))
        self.assertEqual(pairs, [])
        self.assertEqual(stats["rounds"], 0)
        self.assertEqual(self.clock.pauses, [])
        self.assertEqual(stats["errors"], {"ValidationException": 90})
    def test_chunks_that_answer_with_nothing_are_not_retried(self):
        backend = StubBackend(reply="no json lines here")
        pairs, _, stats = self.run_generation(backend)
        self.assertEqual(pairs, [])
        self.assertEqual(stats["rounds"], 0)
        self.assertEqual(backend.calls, 30)
        self.assertEqual(stats["errors"], {})
    def test_a_healthy_run_makes_no_round(self):
        _, _, stats = self.run_generation(StubBackend())
        self.assertEqual((stats["rounds"], self.clock.pauses, stats["errors"]), (0, [], {}))
    def test_each_round_is_logged_with_the_errors(self):
        with self.assertLogs(main.logger, level="INFO") as logs:
            self.run_generation(StubBackend(fail_calls=90))
        lines = [m for m in logs.output if "round 1:" in m]
        self.assertEqual(len(lines), 1)
        self.assertIn("30 of 30 chunks still without questions", lines[0])
        self.assertIn("ServiceUnavailableException=90", lines[0])
class ErrorCodeTest(unittest.TestCase):
    def test_codes(self):
        self.assertEqual(main._error_code(client_error("ThrottlingException")), "ThrottlingException")
        self.assertEqual(main._error_code(ReadTimeoutError(endpoint_url="u")), "Timeout")
        self.assertEqual(main._error_code(ConnectTimeoutError(endpoint_url="u")), "Timeout")
        self.assertEqual(main._error_code(EndpointConnectionError(endpoint_url="u")), "Connection")
        self.assertEqual(main._error_code(ValueError("x")), "Other")
    def test_retryable_set(self):
        for code in ("ServiceUnavailableException", "InternalServerException", "ThrottlingException",
                     "ModelNotReadyException", "AccessDeniedException", "Timeout", "Connection"):
            self.assertIn(code, main.RETRYABLE_CODES)
        self.assertNotIn("ValidationException", main.RETRYABLE_CODES)
        self.assertNotIn("Other", main.RETRYABLE_CODES)
    def test_the_tally_counts_from_many_threads(self):
        tally = main._ErrorTally()
        threads = [threading.Thread(target=lambda: [tally.record("X") for _ in range(500)])
                   for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(tally.snapshot(), {"X": 4000})
    def test_a_failed_attempt_is_counted_in_the_global_tally(self):
        before = main.ERROR_TALLY.snapshot()
        with mock.patch.object(main.time, "sleep", lambda s: None):
            self.assertIsNone(main._call_with_retry(StubBackend(fail_calls=None), "p"))
        delta = main._tally_since(before, main.ERROR_TALLY.snapshot())
        self.assertEqual(delta, {"ServiceUnavailableException": 3})
class S3Stub:
    """Stands for boto3.client("s3") in run_embedding_pipeline."""
    def __init__(self):
        self.puts = {}
        self.deletes = []
    def put_object(self, Bucket, Key, Body):
        self.puts[Key] = Body
    def delete_object(self, Bucket, Key):
        self.deletes.append(Key)
class FloorTest(RoundsBase):
    DATA_CFG = {"bucket": "b", "path": "p/", "sources": []}
    def run_pipeline(self, backend, target_pairs=30, corpus=CORPUS):
        self.s3 = S3Stub()
        self.written = {}
        with mock.patch.object(main, "_select_backend", lambda model, endpoint: backend), \
                mock.patch.object(main.boto3, "client", lambda *a, **k: self.s3), \
                mock.patch.object(main, "write_jsonl_to_s3",
                                  lambda bucket, key, records: self.written.__setitem__(key, records)):
            main.run_embedding_pipeline({}, self.DATA_CFG, "proj", "d", corpus,
                                        {"model": "m", "pairs": target_pairs})
    def pipeline_with_queries(self, per_chunk, target_pairs=30):
        """Every chunk returns `per_chunk` queries; the pipeline asks for 3."""
        reply = "\n".join(json.dumps({"query": "q%d" % i}) for i in range(per_chunk))
        self.run_pipeline(StubBackend(reply=reply), target_pairs)
    def test_the_floor_stops_a_run_after_an_outage_and_names_the_cause(self):
        with self.assertLogs(main.logger, level="INFO") as logs:
            with self.assertRaises(SystemExit) as cm:
                self.run_pipeline(StubBackend(fail_calls=None))
        self.assertEqual(cm.exception.code, 1)
        [msg] = [m for m in logs.output if "Stopping below" in m]
        self.assertIn("Generated 0 of 30 expected pairs from 300 chunks after 5 rounds", msg)
        self.assertIn("ServiceUnavailableException=", msg)
        self.assertIn("primary us-west-2, fallback us-east-1", msg)
        self.assertIn("Stopping below 15 pairs", msg)
        self.assertIn("Check the Bedrock service health for m", msg)
        self.assertNotIn("proj/processed/train.jsonl", self.written)
    def test_the_failure_record_is_written_for_the_build(self):
        with self.assertRaises(SystemExit), self.assertLogs(main.logger, level="INFO"):
            self.run_pipeline(StubBackend(fail_calls=None))
        record = json.loads(self.s3.puts["proj/processed/data-failure.json"])
        self.assertEqual((record["pairs"], record["expected"], record["floor"], record["rounds"]),
                         (0, 30, 15, 5))
        self.assertEqual(record["model"], "m")
        self.assertEqual(record["primaryRegion"], "us-west-2")
        self.assertGreater(record["errors"]["ServiceUnavailableException"], 0)
        self.assertIn("Generated 0 of 30 expected pairs", record["message"])
    def test_an_old_failure_record_is_removed_at_the_start_of_a_run(self):
        self.pipeline_with_queries(3)
        self.assertEqual(self.s3.deletes, ["proj/processed/data-failure.json"])
        self.assertNotIn("proj/processed/data-failure.json", self.s3.puts)
        self.assertIn("proj/processed/train.jsonl", self.written)
    def test_under_the_floor_fails_and_above_it_passes(self):
        # 10 chunks are asked for 3 queries each: 30 expected, floor 15.
        # One query per chunk gives 10 pairs.
        with self.assertRaises(SystemExit), self.assertLogs(main.logger, level="INFO"):
            self.pipeline_with_queries(1)
        # Two per chunk gives 20.
        with self.assertLogs(main.logger, level="INFO"):
            self.pipeline_with_queries(2)
        self.assertIn("proj/processed/train.jsonl", self.written)
    def test_the_floor_is_half_of_the_expected_pairs(self):
        stats = {"expected": 396, "chunks": 132, "rounds": 0, "seconds": 0.0, "errors": {}}
        pairs = [{"query": "q", "positive": "p"}] * 198
        self.assertIsNone(main._pair_floor_failure(pairs, stats, "m"))
        failure = main._pair_floor_failure(pairs[:197], stats, "m")
        self.assertEqual((failure["pairs"], failure["floor"]), (197, 198))
    def test_between_the_floor_and_the_expected_count_it_warns(self):
        with self.assertLogs(main.logger, level="INFO") as logs:
            self.pipeline_with_queries(2)
        warnings = [m for m in logs.output if "WARNING" in m and "expected pairs" in m]
        self.assertEqual(len(warnings), 1)
        self.assertIn("Generated 20 of 30 expected pairs, above the floor of 15", warnings[0])
    def test_a_full_run_does_not_warn(self):
        with self.assertLogs(main.logger, level="INFO") as logs:
            self.pipeline_with_queries(3)
        self.assertEqual([m for m in logs.output if "WARNING" in m], [])
    def test_sources_with_no_chunks_still_fail(self):
        with self.assertRaises(SystemExit) as cm, self.assertLogs(main.logger, level="INFO") as logs:
            self.run_pipeline(StubBackend(), corpus=[{"source": "a", "content": ""}])
        self.assertEqual(cm.exception.code, 1)
        self.assertTrue(any("No (query, positive) pairs generated" in m for m in logs.output))
    def test_a_failure_to_write_the_record_does_not_hide_the_failure(self):
        class Broken(S3Stub):
            def put_object(self, **kw):
                raise RuntimeError("denied")
        self.s3 = Broken()
        with mock.patch.object(main, "_select_backend", lambda m, e: StubBackend(fail_calls=None)), \
                mock.patch.object(main.boto3, "client", lambda *a, **k: self.s3):
            with self.assertRaises(SystemExit) as cm, self.assertLogs(main.logger, level="INFO") as logs:
                main.run_embedding_pipeline({}, self.DATA_CFG, "proj", "d", CORPUS,
                                            {"model": "m", "pairs": 30})
        self.assertEqual(cm.exception.code, 1)
        self.assertTrue(any("Could not write proj/processed/data-failure.json" in m for m in logs.output))
class HeldOutShortfallTest(unittest.TestCase):
    """A held-out set that came back short is reported, once."""
    CONFIG = """
project: {name: triage, domain: d, task: classification}
data:
  bucket: b
  path: p/
  sources: []
  synthetic: {model: m, pairs: 20}
  evaluation: {model: m, pairs: 10}
"""
    def run_main(self, eval_returned):
        calls = []
        def generate(records, model, endpoint, target_pairs, **kw):
            calls.append(target_pairs)
            n = 20 if len(calls) == 1 else eval_returned
            return [{"instruction": "i", "input": "x%d" % i, "output": "a"} for i in range(n)]
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(self.CONFIG)
        self.addCleanup(os.remove, f.name)
        with mock.patch.object(sys, "argv", ["main.py", f.name]), \
                mock.patch.object(main, "read_raw_sources", lambda b, p, s: [{"source": "s", "content": "c"}]), \
                mock.patch.object(main, "read_labeled_records", lambda *a, **k: []), \
                mock.patch.object(main, "generate_synthetic", generate), \
                mock.patch.object(main, "write_jsonl_to_s3", lambda *a: None), \
                mock.patch.object(main, "_write_output_stats", lambda *a: None), \
                self.assertLogs(main.logger, level="INFO") as logs:
            main.main()
        return [m for m in logs.output if "WARNING" in m and "held-out" in m]
    def test_a_short_held_out_set_logs_one_warning(self):
        [line] = self.run_main(eval_returned=6)
        self.assertIn("has 6 of 10 records", line)
    def test_a_full_held_out_set_logs_nothing(self):
        self.assertEqual(self.run_main(eval_returned=10), [])
if __name__ == "__main__":
    unittest.main()
