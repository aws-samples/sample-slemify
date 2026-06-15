# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Shared, dependency-light token-tagger logic for Slemify extraction.
# Pure Python (numpy only in the predict path). Used IDENTICALLY by the trainer
# (fit) and the serving pod (predict), so the feature functions MUST match
# exactly. This file is duplicated verbatim in classifier-serving/ — keep them
# in sync. Features are domain-agnostic (no per-domain gazetteers) so the same
# tagger works for any entity taxonomy the user defines.

import re

TOKEN_RE = re.compile(r"\w+(?:[-./]\w+)*|[^\w\s]")


def tokenize(text):
    """Tokens as (text, start, end), keeping hyphen/dot/slash-joined tokens
    (e.g. checkout-service, v2.3.1) whole."""
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _shape(w):
    s = "".join("X" if c.isupper() else "x" if c.islower()
                else "d" if c.isdigit() else c for c in w)
    return s[:8]


def token_features(tokens, i):
    """Generic NER features for token i with a +/-2 context window. Binary
    features are ints (0/1) so DictVectorizer encodes them as numeric (one
    feature name), which the serving side reproduces exactly."""
    tok = tokens[i][0]
    low = tok.lower()
    f = {
        "w": low, "shape": _shape(tok),
        "pre2": low[:2], "pre3": low[:3], "suf2": low[-2:], "suf3": low[-3:],
        "title": 1 if tok.istitle() else 0,
        "upper": 1 if (tok.isupper() and len(tok) > 1) else 0,
        "hasdig": 1 if any(c.isdigit() for c in tok) else 0,
        "alldig": 1 if tok.isdigit() else 0,
        "hashyf": 1 if "-" in tok else 0,
        "hasdot": 1 if "." in tok else 0,
        "punct": 1 if not any(c.isalnum() for c in tok) else 0,
        "bias": 1,
    }
    if i == 0:
        f["BOS"] = 1
    if i == len(tokens) - 1:
        f["EOS"] = 1
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(tokens):
            f["w%+d" % off] = tokens[j][0].lower()
            f["shape%+d" % off] = _shape(tokens[j][0])
    return f


def gold_char_spans(text, entities):
    """Char spans (start, end, type) for every occurrence of each gold entity
    surface in text. entities: list of {"type":..., "text":...}."""
    spans = []
    for e in entities:
        t, txt = e.get("type"), e.get("text", "")
        if not t or not txt:
            continue
        start = 0
        while True:
            k = text.find(txt, start)
            if k < 0:
                break
            spans.append((k, k + len(txt), t))
            start = k + len(txt)
    return spans


def bio_tags(tokens, spans):
    """Per-token BIO tag from gold char spans (first overlapping span wins)."""
    tags = []
    for tok, s, e in tokens:
        tag = "O"
        for gs, ge, gt in spans:
            if s < ge and e > gs:
                tag = ("B-" if s <= gs else "I-") + gt
                break
        tags.append(tag)
    return tags


def decode_spans(text, tokens, tags):
    """BIO tag sequence -> list of (type, surface) using exact char offsets."""
    out, cur, ctype, cs, ce = [], False, None, 0, 0
    for (tok, s, e), tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if cur:
                out.append((ctype, cs, ce))
            cur, ctype, cs, ce = True, tag[2:], s, e
        elif tag.startswith("I-") and cur and tag[2:] == ctype:
            ce = e
        else:
            if cur:
                out.append((ctype, cs, ce))
            cur = False
    if cur:
        out.append((ctype, cs, ce))
    return [(t, text[a:b]) for (t, a, b) in out]


def predict_tags(text, vocab, coef, intercept, classes):
    """Serving-side inference: reproduce DictVectorizer.transform + sklearn
    LogisticRegression.predict (argmax of the decision function) with numpy, so
    the pod needs no sklearn. coef: [n_classes, n_features]; intercept:
    [n_classes]; classes: tag names aligned to coef rows."""
    import numpy as np
    toks = tokenize(text)
    if not toks:
        return toks, []
    n = coef.shape[1]
    tags = []
    for i in range(len(toks)):
        x = np.zeros(n, dtype=np.float64)
        for k, v in token_features(toks, i).items():
            if isinstance(v, str):
                key, val = "%s=%s" % (k, v), 1.0
            else:
                key, val = k, float(v)
            idx = vocab.get(key)
            if idx is not None:
                x[idx] = val
        scores = coef @ x + intercept
        tags.append(classes[int(np.argmax(scores))])
    return toks, tags


def extract(text, head):
    """Top-level extraction: text + head dict -> [{"type","text"}, ...]."""
    import numpy as np
    coef = np.asarray(head["coef"], dtype=np.float64)
    intercept = np.asarray(head["intercept"], dtype=np.float64)
    toks, tags = predict_tags(text, head["vocab"], coef, intercept, head["classes"])
    spans = decode_spans(text, toks, tags)
    seen, out = set(), []
    for t, s in spans:
        key = (t, s.lower().strip())
        if s.strip() and key not in seen:
            seen.add(key)
            out.append({"type": t, "text": s})
    return out
