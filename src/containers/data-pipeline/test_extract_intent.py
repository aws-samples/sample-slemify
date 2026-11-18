"""Label handling in report.llm_zero_shot_baseline, the only place report.py
reads a label out of text. The expected label is the part of the held-out
output before the first "|". The predicted label is the first label, in label
order, that appears in the frontier model's reply (case-insensitive), else the
reply's first word. Run from this directory:
python3 -m unittest test_extract_intent -v
"""
import os
import sys
import unittest
from unittest import mock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
import report  # noqa: E402

LABELS = ["refund_request", "setup_help", "billing_question"]


def run(rows, replies, labels=LABELS, max_samples=100):
    """Run the baseline with one stubbed reply per row. A reply that is an
    exception is raised by the stub, like a failed Bedrock call."""
    replies = iter(replies)

    def reply(prompt, max_tokens=64):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r
    with mock.patch.object(report, "BEDROCK_MODEL", "model-x"), \
            mock.patch.object(report, "MAX_SAMPLES", max_samples), \
            mock.patch.object(report, "bedrock_text", side_effect=reply), \
            mock.patch.object(report, "p"):
        return report.llm_zero_shot_baseline(rows, labels)


def row(output, origin="synthetic"):
    return {"input": "message text", "output": output, "origin": origin}


class ExpectedLabelTest(unittest.TestCase):
    def test_part_before_the_first_pipe(self):
        out = run([row("refund_request|angry"), row("setup_help | neutral"), row("billing_question")],
                  ["refund_request", "setup_help", "billing_question"])
        self.assertEqual([x["expected"] for x in out["predictions"]],
                         ["refund_request", "setup_help", "billing_question"])
        self.assertEqual(out["accuracy"], 1.0)


class PredictedLabelTest(unittest.TestCase):
    def predicted(self, reply, labels=LABELS):
        return run([row("refund_request|angry")], [reply], labels)["predictions"][0]["predicted"]

    def test_label_found_in_the_reply(self):
        self.assertEqual(self.predicted("Category: Refund_Request."), "refund_request")
        self.assertEqual(self.predicted("setup_help"), "setup_help")

    def test_first_label_in_label_order_wins(self):
        self.assertEqual(self.predicted("setup_help or refund_request"), "refund_request")
        self.assertEqual(self.predicted("refund_request", ["refund", "refund_request"]), "refund")

    def test_no_label_falls_back_to_the_first_word(self):
        self.assertEqual(self.predicted("unsure, maybe a refund"), "unsure,")

    def test_empty_reply(self):
        self.assertEqual(self.predicted(""), "")


class BaselineTest(unittest.TestCase):
    def test_skips_rows_without_input_or_output_and_caps_samples(self):
        rows = [{"input": "", "output": "setup_help"}, {"input": "x"}] + [row("setup_help")] * 3
        # The two bad rows are dropped before the cap applies.
        self.assertEqual(run(rows, ["setup_help"] * 3, max_samples=4)["n"], 3)
        self.assertEqual(run(rows, ["setup_help"] * 2, max_samples=2)["n"], 2)

    def test_failed_call_is_not_counted(self):
        out = run([row("setup_help"), row("setup_help")], [RuntimeError("throttled"), "setup_help"])
        self.assertEqual(out["n"], 1)
        self.assertEqual(out["accuracy"], 1.0)

    def test_accuracy_by_origin(self):
        out = run([row("setup_help", "real"), row("setup_help", "real"), row("billing_question")],
                  ["setup_help", "refund_request", "billing_question"])
        self.assertEqual(out["accuracy"], round(2 / 3, 4))
        self.assertEqual(out["by_origin"]["real"], {"n": 2, "correct": 1, "accuracy": 0.5})
        self.assertEqual(out["by_origin"]["synthetic"]["accuracy"], 1.0)

    def test_nothing_to_run(self):
        self.assertIsNone(run([row("setup_help")], [], labels=[]))
        self.assertIsNone(run([], []))
        with mock.patch.object(report, "BEDROCK_MODEL", ""):
            self.assertIsNone(report.llm_zero_shot_baseline([row("setup_help")], LABELS))


if __name__ == "__main__":
    unittest.main()
