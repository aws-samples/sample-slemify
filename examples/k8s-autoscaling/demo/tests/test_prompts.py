"""The TRIAGE=llm prompt is built from triage/expert.yaml, and project.domain
does not quote or paraphrase any held-out triage message."""
import os
import re
import tempfile
import unittest
from unittest import mock

from tests import _path

from agent import classify, prompts  # noqa: E402

triage_acc = _path.load_script("eval/triage_acc.py", "triage_acc_for_prompts")

# Tokenizer and stopwords copied from the leakage check that cleared the
# boundary rules (domain-rules-eval/scripts/leakage_rules.py).
STOP = set("a an the and or of to in on for is are be it this that with as at by from not only "
           "when also such about how which what if do does".split())


def words(t):
    return re.findall(r"[a-z0-9_\-]+", t.lower())


def ngrams(w, n):
    return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.])\s+", t.strip()) if len(s.strip()) > 3]


class PromptTest(unittest.TestCase):
    def setUp(self):
        prompts.triage_domain.cache_clear()
        self.addCleanup(prompts.triage_domain.cache_clear)

    def test_prompt_has_labels_rules_and_message(self):
        labels, _ = prompts.triage_domain()
        p = prompts.triage_llm_prompt("my HPA will not scale")
        for label in labels:
            self.assertIn(f"  {label}\n", p)
        self.assertIn("A message is not in scope only because it mentions Kubernetes.", p)
        self.assertIn("--- MESSAGE ---\nmy HPA will not scale\n--- END MESSAGE ---", p)
        self.assertIn("label|confidence", p)

    def test_valid_categories_match_routing_labels(self):
        labels, _ = prompts.triage_domain()
        self.assertEqual(classify._VALID_CATEGORIES, set(labels))

    def test_env_config_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "expert.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("project:\n  domain: >\n    Only the test rule\n    applies.\n"
                        "  labels:\n    routing: [alpha, beta]\n")
            with mock.patch.dict(os.environ, {"TRIAGE_EXPERT_CONFIG": path}):
                p = prompts.triage_llm_prompt("x")
        self.assertIn("  alpha\n  beta\n", p)
        self.assertIn("Labeling rules:\nOnly the test rule applies.\n", p)
        self.assertNotIn("karpenter_config\n", p)

    def test_missing_config_raises_on_build(self):
        with mock.patch.dict(os.environ, {"TRIAGE_EXPERT_CONFIG": "/nonexistent/expert.yaml"}):
            with self.assertRaises(FileNotFoundError) as cm:
                prompts.triage_llm_prompt("x")
        self.assertIn("TRIAGE_EXPERT_CONFIG", str(cm.exception))


class LeakageGuardTest(unittest.TestCase):
    """project.domain must not quote or paraphrase a held-out message: no
    shared word 4-gram (stopword-only ones ignored) and no domain sentence
    whose word 3-grams appear in a held-out message."""

    def test_domain_shares_no_ngrams_with_heldout(self):
        prompts.triage_domain.cache_clear()
        _, domain = prompts.triage_domain()
        held = triage_acc.load_set(os.path.join(_path.DEMO_DIR, "eval", "triage-heldout.yaml"))
        self.assertEqual(len(held), 43)
        sents = sentences(domain)
        shared, worst = [], 0.0
        for h in held:
            hw = words(h["query"])
            h4 = {g for g in ngrams(hw, 4) if not all(x in STOP for x in g)}
            h3 = ngrams(hw, 3)
            for s in sents:
                sw = words(s)
                s4 = {g for g in ngrams(sw, 4) if not all(x in STOP for x in g)}
                if h4 & s4:
                    shared.append((h["id"], sorted(" ".join(g) for g in h4 & s4)))
                s3 = ngrams(sw, 3)
                worst = max(worst, len(h3 & s3) / max(len(s3), 1))
        self.assertEqual(shared, [])
        self.assertEqual(worst, 0.0)


if __name__ == "__main__":
    unittest.main()
