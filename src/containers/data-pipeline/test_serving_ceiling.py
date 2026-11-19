"""The bandwidth estimate in report.serving_profile and the cold time-to-first
-token finding in report.generation_report. A model that reads only its active
set per token gets two ends (a floor from the whole file, a ceiling from the
active set); a dense model gets one figure, as before. Run from this directory:
python3 -m unittest test_serving_ceiling -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

import report  # noqa: E402

# 18.55 GB of weights on a node with a 133 GB/s share, decoding at 40.3 tok/s:
# the figures the analyst deploy printed in the end-to-end run.
MODEL_BYTES = 18_550_000_000
DECODE_TOK_S = 40.3


def profile(base_model, decode_tok_s=DECODE_TOK_S, instance="c9g.8xlarge", vcpus=32):
    """Run serving_profile with the endpoint stubbed out."""
    def http_json(path, body=None, timeout=None):
        if path == "/v1/models":
            return {"data": [{"meta": {"size": MODEL_BYTES, "n_params": 30_000_000_000,
                                       "n_ctx_train": 262144}}]}, 1.0
        return {"model_path": "/models/model-q4_k_m.gguf",
                "default_generation_settings": {"n_ctx": 8192}, "total_slots": 1}, 1.0

    def chat(prompt, max_tokens=160, timeout=900):
        return {"timings": {"prompt_n": 922, "prompt_ms": 11161.0, "prompt_per_second": 82.6,
                            "predicted_n": 120, "predicted_ms": 2977.0,
                            "predicted_per_second": decode_tok_s}}, 14000.0

    with mock.patch.object(report, "BASE_MODEL", base_model), \
            mock.patch.object(report, "INSTANCE_TYPE", instance), \
            mock.patch.object(report, "INSTANCE_VCPUS", vcpus), \
            mock.patch.object(report, "http_json", http_json), \
            mock.patch.object(report, "_chat", chat), \
            mock.patch.object(report, "p"):
        return report.serving_profile()


class ActiveParameterFractionTest(unittest.TestCase):
    def test_reads_the_active_share_from_a_mixture_of_experts_id(self):
        self.assertAlmostEqual(report.active_parameter_fraction("Qwen/Qwen3-30B-A3B-Instruct-2507"), 0.1)

    def test_dense_and_unknown_ids_have_no_active_share(self):
        for model_id in ("meta-llama/Llama-3.1-8B-Instruct", "", None, "a-0b-a0b-model"):
            self.assertIsNone(report.active_parameter_fraction(model_id), model_id)


class CeilingTest(unittest.TestCase):
    def test_active_set_sets_the_ceiling_and_the_whole_file_the_floor(self):
        ceil = profile("Qwen/Qwen3-30B-A3B-Instruct-2507")["ceiling"]
        # 1/10 of the file per token, so the ceiling is ten times the floor.
        self.assertEqual(ceil["bytes_per_token_whole_file"], MODEL_BYTES)
        self.assertEqual(ceil["bytes_per_token_assumed"], round(MODEL_BYTES * 0.1))
        self.assertLess(ceil["tokens_per_second_floor"], ceil["tokens_per_second_ceiling"])
        self.assertLess(DECODE_TOK_S, ceil["tokens_per_second_ceiling"])
        self.assertGreater(DECODE_TOK_S, ceil["tokens_per_second_floor"])
        self.assertIn("active set", ceil["assumption"])

    def test_a_measurement_inside_the_range_is_not_reported_as_low(self):
        ceil = profile("Qwen/Qwen3-30B-A3B-Instruct-2507")["ceiling"]
        self.assertIn("inside the estimated range", ceil["reading"])
        self.assertNotIn("well below", ceil["reading"])

    def test_a_dense_model_keeps_one_figure(self):
        ceil = profile("meta-llama/Llama-3.1-8B-Instruct")["ceiling"]
        self.assertEqual(ceil["bytes_per_token_assumed"], MODEL_BYTES)
        self.assertEqual(ceil["tokens_per_second_floor"], ceil["tokens_per_second_ceiling"])
        self.assertGreater(ceil["measured_fraction"], 1)
        self.assertIn("whole model file", ceil["assumption"])

    def test_a_measurement_below_the_floor_still_asks_for_a_check(self):
        ceil = profile("Qwen/Qwen3-30B-A3B-Instruct-2507", decode_tok_s=5.0)["ceiling"]
        self.assertIn("well below", ceil["reading"])


class ColdFindingTest(unittest.TestCase):
    def finding(self):
        with mock.patch.object(report, "INFERENCE", "http://inference"), \
                mock.patch.object(report, "CASES_KEY", ""), \
                mock.patch.object(report, "serving_profile",
                                  lambda: {"cold": {"ttft_ms": 11161.0, "prompt_tokens": 922},
                                           "warm": {"ttft_ms": 51.0, "decode_tok_s": DECODE_TOK_S}}), \
                mock.patch.object(report, "serving_block", lambda: {}), \
                mock.patch.object(report, "p"):
            return report.generation_report()["findings"][0]

    def test_cold_prefill_does_not_point_at_an_accelerator(self):
        text = self.finding()
        self.assertIn("Cold time to first token is 11.2 s", text)
        self.assertNotIn("accelerator", text)

    def test_cold_prefill_says_what_the_number_is(self):
        text = self.finding()
        self.assertIn("prompt processing", text)
        self.assertIn("no cached prefix", text)


if __name__ == "__main__":
    unittest.main()
