# Support Ticket Entity Extraction (task: extraction)

A Slemify `extraction` expert that pulls typed entity spans out of free-form
software support and incident tickets:

- **SERVICE** — affected service/component (`checkout-service`, `the payments backend`)
- **ERROR** — error type or signature (`OOMKilled`, `502 Bad Gateway`)
- **VERSION** — version string (`v2.3.1`, `1.4.0`)
- **ENV** — environment or region (`production`, `us-east-1`)

Extracted entities let you auto-route tickets, deduplicate incidents, and
correlate alerts — on CPU, with no LLM call per ticket.

## Why this domain (and not k8s)

Extraction only earns its keep where entities are **open-vocabulary and live in
prose** — you can't enumerate every service name in a gazetteer or catch every
phrasing with regex. Support tickets fit; structured Kubernetes configs do not
(there a YAML parser already extracts `apiVersion`/`kind` perfectly). We measured
this before adding the task — see the gate results below.

## How it works

`task: extraction` (v1) trains a **feature-based token tagger** — logistic
regression over per-token features (the token, its shape, affixes, casing, and a
±2-word context window) with BIO tagging. It runs entirely on CPU, trains in
seconds, and serves with **no encoder and no ONNX** (the linear model is
reproduced in numpy at serving time). That is why `model.base` is omitted — this
tagger uses no embedding model.

The features are domain-agnostic, so the same pipeline works for any entity
taxonomy you put under `project.labels`.

## What the data pipeline generates

Bedrock writes synthetic tickets and labels the entities in each, stored as
`{instruction, input, output}` where `output` is `TYPE :: surface || TYPE :: surface`
and every surface is an exact substring of the input. The trainer turns those
into per-token BIO labels.

## Measured results

On 280 synthetic tickets (224 train / 56 held-out eval), entity-level span F1:

| Entity | Regex+gazetteer baseline | Trained tagger |
|--------|-------------------------:|---------------:|
| VERSION | 1.00 | 1.00 |
| ENV | 0.92 | 1.00 |
| ERROR | 0.40 | 0.68 |
| SERVICE | 0.21 | 0.94 |
| **Overall** | **0.63** | **0.89** |

The trained tagger wins where it matters: the open-vocabulary SERVICE entity
(recall 0.14 → ~0.95) and ERROR, while matching the baseline on the
pattern-matchable VERSION/ENV. The training report also prints a
**memorize-training-surfaces** baseline (string-matching entity surfaces seen in
training); the tagger beats it too, confirming it generalizes to unseen surfaces
rather than memorizing.

## Run it

```bash
slemify deploy --config expert.yaml
```

This generates the synthetic labeled data, trains the tagger on CPU, serves it,
and prints the extraction report (span P/R/F1 per entity type + baseline).

## Query it

The served model exposes a native endpoint and an OpenAI-compatible one:

```bash
# Native
curl -s localhost:8080/extract -H 'content-type: application/json' \
  -d '{"input": "checkout-service is throwing 502s in production after the v2.3.1 deploy"}'
# -> {"spans": [{"type":"SERVICE","text":"checkout-service"},
#               {"type":"ERROR","text":"502"},
#               {"type":"ENV","text":"production"},
#               {"type":"VERSION","text":"v2.3.1"}]}
```

The chat-completions endpoint returns the same spans as JSON in the message
content, so it drops into an existing orchestrator the same way the classifier
and scorer do.
