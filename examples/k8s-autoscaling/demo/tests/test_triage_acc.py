"""eval/triage_acc.py: errors are retried, then reported apart from misses,
and the cost line matches what the room page parses."""
import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from unittest import mock

from botocore.exceptions import ClientError

from tests import _path

triage_acc = _path.load_script("eval/triage_acc.py", "triage_acc")

# The regexes the room page uses to read a pasted triage-acc output.
ACC_RE = re.compile(r"Triage accuracy \[(llm|classifier)\]:\s*(\d+)\/(\d+)")
LLM_COST_RE = re.compile(r"frontier cost:\s*\$[\d.]+ per call = \$([\d,]+(?:\.\d+)?) per 1M calls")


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": f"{code} for the test"}}, "Converse")


class Calls:
    """A classify.classify stub that replays a script of results/exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.n = 0

    def __call__(self, text):
        self.n += 1
        item = self.script.pop(0) if self.script else {"category": "noise", "confidence": "high"}
        if isinstance(item, Exception):
            raise item
        return item


def no_sleep(_):
    pass


class RetryTest(unittest.TestCase):
    def test_retryable_then_success(self):
        calls = Calls([client_error("ThrottlingException"), {"category": "noise", "confidence": "high"}])
        with mock.patch.object(triage_acc.classify, "classify", calls):
            pred, usd, ms = triage_acc.classify_with_retry("hi", sleep=no_sleep)
        self.assertEqual(pred["category"], "noise")
        self.assertEqual(calls.n, 2)

    def test_retryable_exhausted_raises(self):
        delays = []
        calls = Calls([client_error("ThrottlingException")] * 4)
        with mock.patch.object(triage_acc.classify, "classify", calls):
            with self.assertRaises(ClientError):
                triage_acc.classify_with_retry("hi", sleep=delays.append)
        self.assertEqual(calls.n, 4)
        self.assertEqual(len(delays), 3)
        for d, base in zip(delays, (2, 4, 8)):
            self.assertGreaterEqual(d, base)
            self.assertLessEqual(d, base + 0.5)

    def test_non_retryable_fails_at_once(self):
        calls = Calls([client_error("ValidationException")])
        with mock.patch.object(triage_acc.classify, "classify", calls):
            with self.assertRaises(ClientError):
                triage_acc.classify_with_retry("hi", sleep=no_sleep)
        self.assertEqual(calls.n, 1)


class MainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.set_path = os.path.join(self.tmp.name, "set.yaml")
        with open(self.set_path, "w", encoding="utf-8") as f:
            f.write("- {id: a, label: noise, query: one}\n"
                    "- {id: b, label: noise, query: two}\n"
                    "- {id: c, label: hpa_config, query: three}\n")
        self.results = os.path.join(self.tmp.name, "results")

    def run_main(self, step, script):
        calls = Calls(script)
        out = io.StringIO()
        with mock.patch.object(triage_acc.classify, "classify", calls), \
                mock.patch.object(triage_acc.config, "TRIAGE", step), \
                mock.patch.object(triage_acc.time, "sleep", no_sleep), \
                contextlib.redirect_stdout(out):
            triage_acc.main(["--set", self.set_path, "--results-dir", self.results])
        return out.getvalue()

    def result_json(self):
        (name,) = [n for n in os.listdir(self.results) if n.startswith("triage-")]
        with open(os.path.join(self.results, name), encoding="utf-8") as f:
            return json.load(f)

    def test_error_row_is_not_a_miss(self):
        ok = {"category": "noise", "confidence": "high"}
        out = self.run_main("llm", [ok, client_error("ValidationException"),
                                    {"category": "hpa_config", "confidence": "high"}])
        self.assertIn("[ERR ] b", out)
        self.assertNotIn("[MISS]", out)
        m = ACC_RE.search(out)
        self.assertIsNotNone(m)
        self.assertEqual((m.group(2), m.group(3)), ("2", "2"))
        self.assertIn("(1 not scored: errors) ===", out)
        self.assertIn("1 message(s) failed after 4 attempts and were not scored. "
                      "Run make triage-acc again to score all 3.", out)
        self.assertNotIn("confusions", out)
        res = self.result_json()
        self.assertEqual((res["correct"], res["scored"], res["errors"], res["total"]), (2, 2, 1, 3))
        self.assertEqual(res["accuracy"], 1.0)
        self.assertIn("ValidationException", res["rows"][1]["error"])

    def test_llm_cost_line_unchanged(self):
        real_open = triage_acc.metrics.open_meter

        def open_meter():
            m = real_open()
            m.summary = lambda: {"usd": 0.0012}
            return m

        with mock.patch.object(triage_acc.metrics, "open_meter", open_meter):
            out = self.run_main("llm", [])
        m = LLM_COST_RE.search(out)
        self.assertIsNotNone(m, out)
        self.assertEqual(m.group(1), "1,200")
        self.assertNotIn("CPU cost", out)

    def test_classifier_cost_pointer_without_bench(self):
        out = self.run_main("classifier", [])
        self.assertIn("CPU cost: no per-call bill; the classifier runs on a CPU node billed by the hour. "
                      "make bench-triage prices it per 1M classifications.", out)
        self.assertNotIn("frontier cost", out)

    def test_classifier_cost_from_bench(self):
        os.makedirs(self.results)
        for stamp, usd in (("20250101T000000Z", 9.99), ("20250201T000000Z", 0.42)):
            with open(os.path.join(self.results, f"bench-triage-{stamp}.json"), "w") as f:
                json.dump({"usd_per_1m": usd, "instance_type": "c7i.xlarge"}, f)
        out = self.run_main("classifier", [])
        self.assertIn("CPU cost: $0.42 per 1M classifications on c7i.xlarge "
                      "(make bench-triage, 20250201T000000Z)", out)


if __name__ == "__main__":
    unittest.main()
