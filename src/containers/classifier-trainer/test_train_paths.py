"""Tests for the baked encoder lookup in train.py."""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

import train  # noqa: E402

DEFAULT = "BAAI/bge-base-en-v1.5"


class ResolveModelTest(unittest.TestCase):
    def test_default_name_with_dir_present_returns_dir(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(train.BAKED_ENCODERS, {DEFAULT: d}):
                self.assertEqual(train.resolve_model(DEFAULT), d)

    def test_dir_missing_returns_name(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "absent")
            with mock.patch.dict(train.BAKED_ENCODERS, {DEFAULT: missing}):
                self.assertEqual(train.resolve_model(DEFAULT), DEFAULT)

    def test_other_name_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(train.BAKED_ENCODERS, {DEFAULT: d}):
                self.assertEqual(train.resolve_model("org/other-model"), "org/other-model")


if __name__ == "__main__":
    unittest.main()
