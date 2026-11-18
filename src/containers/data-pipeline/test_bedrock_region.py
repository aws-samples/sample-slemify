"""main._get_bedrock_rpm_quota reads quotas in the primary Bedrock region and
logs a warning when it cannot. Run from this directory:
python3 -m unittest test_bedrock_region -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

import main  # noqa: E402


class QuotaRegionTest(unittest.TestCase):
    def test_quota_client_uses_bedrock_region(self):
        client = mock.MagicMock()
        client.get_paginator.return_value.paginate.return_value = []
        with mock.patch.dict(os.environ, {"BEDROCK_REGION": "us-east-2"}), \
                mock.patch.object(main.boto3, "client", return_value=client) as make:
            self.assertEqual(main._get_bedrock_rpm_quota("anthropic.claude-x"), 0)
        make.assert_called_once()
        self.assertEqual(make.call_args.kwargs["region_name"], "us-east-2")

    def test_quota_error_logs_warning_and_returns_zero(self):
        with mock.patch.dict(os.environ, {"BEDROCK_REGION": "us-east-2"}), \
                mock.patch.object(main.boto3, "client", side_effect=RuntimeError("denied")), \
                self.assertLogs(main.logger, level="WARNING") as logs:
            self.assertEqual(main._get_bedrock_rpm_quota("anthropic.claude-x"), 0)
        self.assertIn("us-east-2", logs.output[0])


if __name__ == "__main__":
    unittest.main()
