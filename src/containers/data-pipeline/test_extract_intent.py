"""Tests for _extract_label — validates label extraction from any output format.

Generic tests — not tied to 'intent' as a field name. Works for any
classification domain (severity, category, priority, etc.).
"""

import sys
sys.path.insert(0, ".")
from report import _extract_label

cases = [
    # Pipe-delimited
    ("refund_request|angry", "refund_request"),
    ("high_severity|critical", "high_severity"),
    ("p1_incident|database", "p1_incident"),

    # JSON — first value with underscore preferred
    ('{"intent": "refund_request", "sentiment": "angry"}', "refund_request"),
    ('{"category": "billing_question", "priority": "high"}', "billing_question"),
    ('{"severity": "high_priority", "source": "api"}', "high_priority"),

    # Markdown table
    ("|intent|sentiment|\n|---|---|\n|feedback|satisfied|", "feedback"),
    ("|category|priority|\n|---|---|\n|bug_report|high|", "bug_report"),
    ("|intent|sentiment|\n|---|---|\n|refund_request|angry|\n\nNote: OCR.", "refund_request"),

    # Key-value with pipes
    ("|intent|: refund_request\n|sentiment|: angry", "refund_request"),
    ("|category|: setup_help\n|urgency|: high", "setup_help"),
    ("|intent|: feedback|sentiment|: frustrated|\n|order|: CBX-123", "feedback"),

    # Colon-pipe format
    ("|intent|:|refund_request|\n|sentiment|:|angry|", "refund_request"),
    ("|intent|:|setup_help|\n|sentiment|:|neutral|", "setup_help"),

    # Key-value without pipes
    ("Intent: refund_request\nSentiment: angry", "refund_request"),
    ("category: billing_question, priority: high", "billing_question"),
    ("severity: critical_outage", "critical_outage"),

    # Expected format (eval data)
    ("setup_help|angry", "setup_help"),
    ("billing_question|neutral", "billing_question"),
    ("technical_issue|frustrated", "technical_issue"),

    # Edge cases
    ("refund_request", "refund_request"),
    ("", ""),
]

passed = 0
failed = 0
for inp, expected in cases:
    got = _extract_label(inp)
    if got == expected:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: expected={expected!r}, got={got!r}")
        print(f"  input={inp[:80]!r}")
        print()

print(f"\n{passed}/{passed+failed} passed, {failed} failed")
sys.exit(1 if failed else 0)
