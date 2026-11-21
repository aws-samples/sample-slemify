"""eval/recall.py: the hosted (bedrock) row retries a transient Bedrock error on
the embedding call instead of crashing `make recall`. No test calls AWS."""
import io
import json
import unittest
from unittest import mock

from botocore.exceptions import ClientError, ReadTimeoutError

from tests import _path
from agent import config

recall = _path.load_script("eval/recall.py", "recall")

OS_HITS = {"hits": {"hits": [{"_source": {"text": "t", "source": "s", "section": "x"}}]}}


def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "InvokeModel")


def titan_ok():
    body = io.BytesIO(json.dumps({"embedding": [0.1, 0.2], "inputTextTokenCount": 3}).encode())
    return {"body": body}


def run_candidates(errors):
    """Run the bedrock candidate pool with a Bedrock client that raises each
    error in turn, then succeeds. Returns (pool, invoke_model mock, sleep mock)."""
    invoke = mock.Mock(side_effect=list(errors) + [titan_ok()])
    fake_bedrock = mock.Mock(invoke_model=invoke)
    fake_os = mock.Mock(search=mock.Mock(return_value=OS_HITS))
    with mock.patch.object(config, "bedrock", fake_bedrock), \
            mock.patch.object(config, "opensearch", fake_os), \
            mock.patch.object(recall.time, "sleep") as sleep, \
            mock.patch("sys.stdout", new_callable=io.StringIO) as out:
        pool = recall.candidates("bedrock", "what is minValues")
    return pool, invoke, sleep, out.getvalue()


class BedrockEmbedRetryTest(unittest.TestCase):
    def test_model_error_once_then_success(self):
        pool, invoke, sleep, out = run_candidates([client_error("ModelErrorException")])
        self.assertEqual(len(pool), 1)
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(out.count("\n"), 1)
        self.assertIn("ModelErrorException", out)

    def test_throttling_once_then_success(self):
        pool, invoke, sleep, out = run_candidates([client_error("ThrottlingException")])
        self.assertEqual(len(pool), 1)
        self.assertEqual(invoke.call_count, 2)

    def test_read_timeout_then_success(self):
        pool, invoke, sleep, out = run_candidates([ReadTimeoutError(endpoint_url="https://bedrock")])
        self.assertEqual(len(pool), 1)
        self.assertEqual(invoke.call_count, 2)

    def test_backoff_grows_and_total_wait_is_bounded(self):
        errors = [client_error("ModelErrorException")] * 3
        _, invoke, sleep, _ = run_candidates(errors)
        waits = [c.args[0] for c in sleep.call_args_list]
        self.assertEqual(len(waits), 3)
        self.assertEqual(sorted(waits), waits)
        self.assertLess(sum(waits), 10)

    def test_gives_up_after_the_last_attempt(self):
        errors = [client_error("ModelErrorException")] * recall.EMBED_ATTEMPTS
        with self.assertRaises(ClientError):
            run_candidates(errors)

    def test_permanent_error_is_not_retried(self):
        invoke = mock.Mock(side_effect=client_error("ValidationException"))
        with mock.patch.object(config, "bedrock", mock.Mock(invoke_model=invoke)), \
                mock.patch.object(recall.time, "sleep") as sleep:
            with self.assertRaises(ClientError):
                recall.candidates("bedrock", "q")
        self.assertEqual(invoke.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
