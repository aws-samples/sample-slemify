"""Tests for the optional revision pin and upload key in convert.py."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convert  # noqa: E402

REV = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
BASE_ENV = {
    "BASE_MODEL": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "QUANTIZE": "q4_k_m",
    "S3_BUCKET": "model-bucket",
    "PROJECT": "k8s-autoscaling-analyst",
    "GGUF_FILENAME": "model-q4_k_m.gguf",
}


class ConvertMainTest(unittest.TestCase):
    def run_main(self, extra):
        env = dict(BASE_ENV)
        env.update(extra)
        for name in ("BASE_MODEL_REVISION", "S3_KEY"):
            env.setdefault(name, "")
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(convert, "snapshot_download") as download, \
                mock.patch.object(convert, "run"), \
                mock.patch.object(convert, "boto3") as boto3, \
                mock.patch("os.path.getsize", return_value=1048576), \
                mock.patch("os.makedirs"):
            upload = boto3.client.return_value.upload_file
            try:
                convert.main()
            finally:
                self.download = download
                self.upload = upload

    def test_revision_passed_when_set(self):
        self.run_main({"BASE_MODEL_REVISION": REV})
        self.assertEqual(self.download.call_args.kwargs["revision"], REV)

    def test_revision_none_and_default_key_when_unset(self):
        self.run_main({})
        self.assertIsNone(self.download.call_args.kwargs["revision"])
        key = self.upload.call_args.args[2]
        self.assertEqual(key, "models/k8s-autoscaling-analyst/model-q4_k_m.gguf")

    def test_invalid_revision_exits_before_download(self):
        with self.assertRaises(SystemExit):
            self.run_main({"BASE_MODEL_REVISION": "main"})
        self.download.assert_not_called()

    def test_invalid_s3_key_exits_before_download(self):
        with self.assertRaises(SystemExit):
            self.run_main({"S3_KEY": "prefix/../$(x)/model-q4_k_m.gguf"})
        self.download.assert_not_called()

    def test_s3_key_used_as_upload_key(self):
        key = "event-1/models/k8s-autoscaling-analyst/model-q4_k_m.gguf"
        self.run_main({"S3_KEY": key})
        self.assertEqual(self.upload.call_args.args[1], "model-bucket")
        self.assertEqual(self.upload.call_args.args[2], key)

    def test_s3_key_not_ending_in_filename_exits(self):
        with self.assertRaises(SystemExit):
            self.run_main({"S3_KEY": "models/k8s-autoscaling-analyst/other.gguf"})
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
