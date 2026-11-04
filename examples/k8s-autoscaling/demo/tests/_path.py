"""Put the demo dir on sys.path so tests can `from agent import ...`, and
load the eval scripts (plain scripts, not a package) by file path.

agent/config.py creates a boto3 client at import, which needs a region. No
test calls AWS.
"""
import importlib.util
import os
import sys

os.environ.setdefault("AWS_REGION", "us-west-2")
os.environ.setdefault("AWS_DEFAULT_REGION", os.environ["AWS_REGION"])

DEMO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)


def load_script(relpath: str, name: str):
    """Import a demo script (for example eval/triage_acc.py) as a module."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(DEMO_DIR, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
