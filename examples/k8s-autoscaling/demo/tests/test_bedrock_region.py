"""bedrock_region.py: calls go to the primary region and fall back to the
second region only on endpoint, access and service errors."""
import filecmp
import os
import re
import unittest
from unittest import mock

from botocore.exceptions import ClientError, EndpointConnectionError

from tests import _path
import bedrock_region

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_path.DEMO_DIR)))
PIPELINE_COPY = os.path.join(REPO_ROOT, "src", "containers", "data-pipeline", "bedrock_region.py")


def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "Converse")


class StubClient:
    """A bedrock-runtime client that raises the given error or answers."""

    def __init__(self, region, error=None):
        self.region = region
        self.error = error
        self.calls = 0

    def converse(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"region": self.region}


class BedrockRegionTest(unittest.TestCase):
    def run_call(self, primary_error, primary="us-west-2", fallback="us-east-1"):
        stubs = {primary: StubClient(primary, primary_error)}
        if fallback != primary:
            stubs[fallback] = StubClient(fallback)

        def make_client(service, region_name=None, config=None):
            self.assertEqual(service, "bedrock-runtime")
            return stubs[region_name]

        env = {"BEDROCK_REGION": primary, "BEDROCK_FALLBACK_REGION": fallback}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(bedrock_region.boto3, "client", side_effect=make_client):
            client = bedrock_region.bedrock_runtime()
            try:
                return client.converse(modelId="m"), stubs
            except Exception as err:
                return err, stubs

    def test_defaults(self):
        with mock.patch.dict(os.environ, {"BEDROCK_REGION": "", "BEDROCK_FALLBACK_REGION": ""}):
            self.assertEqual(bedrock_region.primary_region(), "us-west-2")
            self.assertEqual(bedrock_region.fallback_region(), "us-east-1")

    def test_endpoint_error_falls_back(self):
        got, stubs = self.run_call(EndpointConnectionError(endpoint_url="https://x"))
        self.assertEqual(got, {"region": "us-east-1"})
        self.assertEqual(stubs["us-east-1"].calls, 1)

    def test_access_denied_falls_back(self):
        got, _ = self.run_call(client_error("AccessDeniedException"))
        self.assertEqual(got, {"region": "us-east-1"})

    def test_throttling_raises_without_fallback(self):
        got, stubs = self.run_call(client_error("ThrottlingException"))
        self.assertIsInstance(got, ClientError)
        self.assertEqual(stubs["us-east-1"].calls, 0)

    def test_validation_raises(self):
        got, stubs = self.run_call(client_error("ValidationException"))
        self.assertIsInstance(got, ClientError)
        self.assertEqual(stubs["us-east-1"].calls, 0)

    def test_same_region_has_no_second_call(self):
        got, stubs = self.run_call(client_error("ServiceUnavailableException"),
                                   primary="us-east-1", fallback="us-east-1")
        self.assertIsInstance(got, ClientError)
        self.assertEqual(stubs["us-east-1"].calls, 1)

    def test_fallback_logs_one_warning_with_region(self):
        with self.assertLogs("bedrock_region", level="WARNING") as logs:
            self.run_call(client_error("InternalServerException"))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("us-west-2", logs.output[0])
        self.assertIn("converse", logs.output[0])

    def test_pipeline_copy_is_identical(self):
        self.assertTrue(filecmp.cmp(os.path.join(_path.DEMO_DIR, "bedrock_region.py"),
                                    PIPELINE_COPY, shallow=False))

    def test_no_direct_bedrock_runtime_clients(self):
        pattern = re.compile(r"""boto3\.client\(\s*["']bedrock-runtime""")
        found = []
        for top in (_path.DEMO_DIR, os.path.dirname(PIPELINE_COPY)):
            for root, _, files in os.walk(top):
                for name in files:
                    if name.endswith(".py") and name != "bedrock_region.py" \
                            and not name.startswith("test_"):
                        path = os.path.join(root, name)
                        with open(path, encoding="utf-8") as f:
                            if pattern.search(f.read()):
                                found.append(path)
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
