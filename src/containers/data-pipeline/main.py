# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Slemify Data Pipeline — read raw data, generate training pairs via Bedrock.

Users provide raw, unlabeled data (emails, logs, documents) in S3.
Bedrock generates labeled training pairs from that raw data.

Supports three output shapes:
- pipe_delimited (default): Structured label output (e.g., "label1|label2").
- free_form: Structured reasoning output for audit/analysis tasks.
- scoring: A single numeric target in [0,1] for regression/guardrail tasks.
"""

import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("slemify.data-pipeline")

BATCH_SIZE = 5

GENERATION_PROMPT = """You are generating synthetic training data for a Small Language Model.

DOMAIN CONTEXT:
{domain}

TOOL DESCRIPTION:
{tool_description}

SOURCE EXAMPLES (real data — use as style/content reference):
{source_samples}

VALID OUTPUT LABELS (use ONLY these exact values — no synonyms, no variations):
{valid_labels}

TASK:
Generate exactly {batch_size} training examples.
Each example is a JSON object on its own line (JSONL format) with three fields:
- "instruction": A short, consistent task instruction.
- "input": A realistic input reflecting the domain's real-world messiness.
- "output": The correct response in pipe-delimited format using ONLY the valid labels above. Use the pipe character to separate values. Never use JSON in the output field.

Vary scenarios, writing styles, noise levels, and personas.
Some inputs should be clean, some messy, some extremely noisy.
Keep each input under 150 words.

Output one JSON object per line. No array brackets, no markdown, no explanation:
{{"instruction": "...", "input": "...", "output": "label1|label2"}}
{{"instruction": "...", "input": "...", "output": "label1|label2"}}"""

FREEFORM_GENERATION_PROMPT = """You are generating synthetic training data for a Small Language Model that produces structured reasoning output.

DOMAIN CONTEXT:
{domain}

TOOL DESCRIPTION:
{tool_description}

SOURCE EXAMPLES (real data — use as style/content reference):
{source_samples}

OUTPUT STRUCTURE GUIDELINES:
{output_guidelines}

TASK:
Generate exactly {batch_size} training examples.
Each example is a JSON object on its own line (JSONL format) with three fields:
- "instruction": A short, consistent task instruction describing the analysis task.
- "input": A realistic input reflecting the domain's real-world messiness. Include YAML configs, conversational context, and technical questions. Inputs can be long (up to 500 words).
- "output": A structured reasoning response. Use the output structure guidelines above. The response should include: identification of the issue, explanation of why it's wrong, the correct approach, and risk assessment. Keep outputs between 100-300 words.

Vary scenarios, complexity levels, and error types.
Some inputs should have multiple issues, some just one, some should be valid configs.
Include realistic conversational noise (Slack-style messages, "an LLM told me", etc.).

Output one JSON object per line. No array brackets, no markdown, no explanation:
{{"instruction": "...", "input": "...", "output": "..."}}
{{"instruction": "...", "input": "...", "output": "..."}}"""


SCORING_GENERATION_PROMPT = """You are generating synthetic training data for a Small Language Model that produces a single numeric score.

DOMAIN CONTEXT:
{domain}

TOOL DESCRIPTION:
{tool_description}

SOURCE EXAMPLES (real data — use as style/content reference):
{source_samples}

SCORING GUIDELINES:
{scoring_guidelines}

TASK:
Generate exactly {batch_size} training examples.
Each example is a JSON object on its own line (JSONL format) with three fields:
- "instruction": A short, consistent task instruction describing the scoring task.
- "input": A realistic input reflecting the domain's real-world messiness (configs, questions, messages). Keep each input under 200 words.
- "output": A SINGLE decimal number between 0.0 and 1.0 (inclusive) representing the score for this input. Output ONLY the number — no labels, no words, no JSON, no percent sign.

Spread the scores across the full 0.0–1.0 range: include clearly low cases (near 0.0), clearly high cases (near 1.0), and ambiguous middle cases (around 0.4–0.6). Do not cluster every example near the same value.

Output one JSON object per line. No array brackets, no markdown, no explanation:
{{"instruction": "...", "input": "...", "output": "0.12"}}
{{"instruction": "...", "input": "...", "output": "0.87"}}"""


EMBEDDING_QUERY_PROMPT = """You are generating training data for a domain-tuned text embedding model used in retrieval (RAG).

DOMAIN CONTEXT:
{domain}

You are given a DOCUMENT CHUNK from the knowledge base. Write {n_queries} realistic, diverse questions that this specific chunk answers well. Each question must be answerable from THIS chunk — a retrieval system should return this chunk for that question.

DOCUMENT CHUNK:
{chunk}

Guidelines:
- Write natural questions a practitioner would actually type or ask in chat.
- Vary phrasing: some keyword-style, some full sentences, some with typos or noise.
- Do NOT copy sentences verbatim from the chunk; ask about what it explains.
- Keep each question under 30 words. Do not number them.

Output exactly {n_queries} JSON objects, one per line (JSONL). No array brackets, no markdown:
{{"query": "..."}}
{{"query": "..."}}"""


# === Pipeline ===

def main():
    config_path = "/config/expert.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    logger.info("Loading config from %s", config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    project_name = config["project"]["name"]
    domain = config["project"]["domain"]
    labels_config = config.get("project", {}).get("labels")
    task = config.get("project", {}).get("task", "generation")
    output_format = config.get("project", {}).get("output_format", "")

    # Determine the data shape for synthetic generation and validation:
    #   - free_form: prose/reasoning output (only task=generation + output_format=free_form)
    #   - scoring:   a single numeric target in [0,1] (task=scoring, regression head)
    #   - labels:    structured label output, pipe-separated (classification and
    #                any generation expert not marked free_form)
    # The encoder-head classification path consumes the same label-shaped data
    # as the legacy pipe-delimited generation path; only the trainer differs.
    is_free_form = (task == "generation" and output_format == "free_form")
    is_scoring = (task == "scoring")
    is_embedding = (task == "embedding")
    if is_free_form:
        gen_format = "free_form"
    elif is_scoring:
        gen_format = "scoring"
    else:
        gen_format = "pipe_delimited"
    synthetic_cfg = data_cfg.get("synthetic", {})

    # Phase 1: Read raw source files from S3
    raw_content = read_raw_sources(
        data_cfg["bucket"], data_cfg["path"], data_cfg.get("sources", []))
    logger.info("Read %d raw source files", len(raw_content))

    # Embedding (contrastive) has a different data shape — (query, positive)
    # pairs mined from the document corpus, not {instruction, input, output}
    # records — so it runs a dedicated pipeline and returns early.
    if is_embedding:
        run_embedding_pipeline(config, data_cfg, project_name, domain,
                               raw_content, synthetic_cfg)
        return

    # Phase 2: Generate training pairs via Bedrock
    if not synthetic_cfg.get("model") or not synthetic_cfg.get("pairs", 0):
        logger.error("synthetic.model and synthetic.pairs are required")
        sys.exit(1)

    records = generate_synthetic(
        records=raw_content,
        model=synthetic_cfg["model"],
        endpoint=synthetic_cfg.get("endpoint", ""),
        target_pairs=synthetic_cfg["pairs"],
        domain=domain,
        tools=config.get("project", {}).get("domain", ""),
        labels=labels_config,
        output_format=gen_format,
    )
    logger.info("Generated %d training records", len(records))

    if not records:
        logger.error("No training records generated")
        sys.exit(1)

    # Validate output format
    if is_free_form:
        # For free-form, only drop empty outputs
        valid = [r for r in records if r.get("output", "").strip()]
        dropped = len(records) - len(valid)
        if dropped:
            logger.warning("Dropped %d records with empty output", dropped)
    elif is_scoring:
        # For scoring, keep only records whose output parses as a number in [0,1].
        valid = _validate_scoring(records)
        dropped = len(records) - len(valid)
        if dropped:
            logger.warning("Dropped %d records with non-numeric or out-of-range score", dropped)
    else:
        # For label output, require a non-empty label. Single-dimension
        # classification emits a bare label (no pipe); multi-dimension emits
        # pipe-separated labels. Both are valid as long as output is non-empty.
        valid = [r for r in records if r.get("output", "").strip()]
        dropped = len(records) - len(valid)
        if dropped:
            logger.warning("Dropped %d records with empty label output", dropped)
    records = valid

    if not records:
        logger.error("No valid records after validation")
        sys.exit(1)

    # Check label distribution (only meaningful for label output)
    if not is_free_form and not is_scoring:
        _check_label_balance(records, labels_config)
    elif is_scoring:
        _check_score_distribution(records)

    # Phase 3: Generate eval data and write to S3
    bucket = data_cfg["bucket"]
    eval_cfg = data_cfg.get("evaluation") or {}

    if eval_cfg.get("model") and eval_cfg.get("pairs", 0):
        # Independent eval generation: different model, optionally different source data
        train_records = records  # all synthetic records go to training
        logger.info("Generating independent eval data with %s (%d pairs)...",
                     eval_cfg["model"], eval_cfg["pairs"])

        # Read eval-specific source data if configured, otherwise reuse training sources
        eval_sources = eval_cfg.get("sources", data_cfg.get("sources", []))
        eval_raw = read_raw_sources(data_cfg["bucket"], data_cfg["path"], eval_sources)
        if not eval_raw:
            eval_raw = raw_content  # fallback to training sources
        logger.info("Eval source files: %d", len(eval_raw))

        eval_records = generate_synthetic(
            records=eval_raw,
            model=eval_cfg["model"],
            endpoint="",
            target_pairs=eval_cfg["pairs"],
            domain=domain,
            tools=config.get("project", {}).get("domain", ""),
            labels=labels_config,
            output_format=gen_format,
        )
        # Both free-form and label output are valid when non-empty.
        if is_scoring:
            eval_valid = _validate_scoring(eval_records)
        else:
            eval_valid = [r for r in eval_records if r.get("output", "").strip()]
        eval_dropped = len(eval_records) - len(eval_valid)
        if eval_dropped:
            logger.warning("Dropped %d eval records with empty output", eval_dropped)
        eval_records = eval_valid
        logger.info("Generated %d independent eval records", len(eval_records))
    else:
        # Fallback: split training data into train/eval
        split_ratio = data_cfg.get("split_ratio", 0.9)
        split_idx = int(len(records) * split_ratio)
        train_records = records[:split_idx]
        eval_records = records[split_idx:]

    logger.info("Writing %d train / %d eval records to s3://%s/",
                len(train_records), len(eval_records), bucket)
    write_jsonl_to_s3(bucket, f"{project_name}/processed/train.jsonl", train_records)
    write_jsonl_to_s3(bucket, f"{project_name}/processed/eval.jsonl", eval_records)

    # Compute and store output token stats for serving configuration
    _write_output_stats(bucket, project_name, train_records)

    logger.info("Data pipeline complete")


# === Embedding (contrastive) pipeline ===

def _chunk_documents(raw_content, chunk_chars=1200, overlap=150, min_chars=200):
    """Split raw source files into retrieval-sized chunks.

    Each chunk becomes a "positive" document; Bedrock writes the queries it
    answers. Chunks are paragraph-aware (split on blank lines) and then packed
    to roughly chunk_chars so they fit comfortably under the encoder's token cap.
    """
    chunks = []
    for rec in raw_content:
        content = rec.get("content", "").strip()
        source = rec.get("source", "unknown")
        if len(content) < min_chars:
            if content:
                chunks.append({"text": content, "source": source})
            continue
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 > chunk_chars and len(buf) >= min_chars:
                chunks.append({"text": buf.strip(), "source": source})
                buf = buf[-overlap:] + "\n\n" + para
            else:
                buf = (buf + "\n\n" + para) if buf else para
        if len(buf.strip()) >= min_chars:
            chunks.append({"text": buf.strip(), "source": source})
    return chunks


def _generate_queries_for_chunk(backend, domain, chunk_text, n_queries):
    """Ask the backend for n_queries questions answerable by this chunk."""
    prompt = EMBEDDING_QUERY_PROMPT.format(
        domain=domain, chunk=chunk_text[:4000], n_queries=n_queries)
    response = _call_with_retry(backend, prompt)
    if not response:
        return []
    queries = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            q = str(obj.get("query", "")).strip()
            if q:
                queries.append(q)
        except json.JSONDecodeError:
            continue
    return queries


def _generate_pairs(raw_content, model, endpoint, target_pairs, domain, queries_per_chunk=3):
    """Generate (query, positive) pairs by writing queries for each doc chunk."""
    backend = _select_backend(model, endpoint)
    concurrency = _calculate_concurrency(model) if not endpoint else 20

    chunks = _chunk_documents(raw_content)
    if not chunks:
        return [], []
    logger.info("Chunked corpus into %d document chunks", len(chunks))

    # Only process as many chunks as needed to hit the target pair count.
    needed_chunks = min(len(chunks), (target_pairs + queries_per_chunk - 1) // queries_per_chunk)
    selected = random.sample(chunks, needed_chunks)

    pairs = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_generate_queries_for_chunk, backend, domain,
                        c["text"], queries_per_chunk): c
            for c in selected
        }
        done = 0
        for future in as_completed(futures):
            chunk = futures[future]
            done += 1
            try:
                queries = future.result()
            except Exception as e:
                logger.warning("Query generation failed: %s", e)
                continue
            for q in queries:
                pairs.append({"query": q, "positive": chunk["text"],
                              "source": chunk.get("source", "")})
            if done % 25 == 0:
                logger.info("  %d/%d chunks processed, %d pairs so far",
                            done, len(selected), len(pairs))

    # Return both the pairs and the full chunk corpus (used as the retrieval
    # index for eval recall@k).
    return pairs[:target_pairs], chunks


def run_embedding_pipeline(config, data_cfg, project_name, domain, raw_content, synthetic_cfg):
    """Generate contrastive (query, positive) pairs and a retrieval corpus.

    Writes:
      <project>/processed/train.jsonl  — {query, positive} pairs for training
      <project>/processed/eval.jsonl   — {query, positive, source} held-out pairs
      <project>/processed/corpus.jsonl — {text, source} all chunks (eval index)
    """
    if not synthetic_cfg.get("model") or not synthetic_cfg.get("pairs", 0):
        logger.error("synthetic.model and synthetic.pairs are required")
        sys.exit(1)
    if not raw_content:
        logger.error("task=embedding requires source documents to mine pairs from")
        sys.exit(1)

    bucket = data_cfg["bucket"]
    logger.info("Generating %d (query, positive) pairs via %s...",
                synthetic_cfg["pairs"], synthetic_cfg["model"])

    pairs, chunks = _generate_pairs(
        raw_content=raw_content,
        model=synthetic_cfg["model"],
        endpoint=synthetic_cfg.get("endpoint", ""),
        target_pairs=synthetic_cfg["pairs"],
        domain=domain,
    )
    if not pairs:
        logger.error("No (query, positive) pairs generated")
        sys.exit(1)
    logger.info("Generated %d pairs from %d chunks", len(pairs), len(chunks))

    # Independent eval: prefer a separate eval source/model when configured,
    # otherwise hold out a slice of the generated pairs.
    eval_cfg = data_cfg.get("evaluation") or {}
    if eval_cfg.get("model") and eval_cfg.get("pairs", 0):
        eval_sources = eval_cfg.get("sources", data_cfg.get("sources", []))
        eval_raw = read_raw_sources(bucket, data_cfg["path"], eval_sources) or raw_content
        eval_pairs, _ = _generate_pairs(
            raw_content=eval_raw, model=eval_cfg["model"], endpoint="",
            target_pairs=eval_cfg["pairs"], domain=domain)
        train_pairs = pairs
    else:
        split_ratio = data_cfg.get("split_ratio", 0.9)
        split_idx = int(len(pairs) * split_ratio)
        train_pairs = pairs[:split_idx]
        eval_pairs = pairs[split_idx:]

    if not eval_pairs:
        # Guarantee a non-empty eval set even for tiny corpora.
        eval_pairs = train_pairs[-max(1, len(train_pairs) // 10):]

    logger.info("Writing %d train / %d eval pairs and %d-chunk corpus to s3://%s/",
                len(train_pairs), len(eval_pairs), len(chunks), bucket)
    write_jsonl_to_s3(bucket, f"{project_name}/processed/train.jsonl", train_pairs)
    write_jsonl_to_s3(bucket, f"{project_name}/processed/eval.jsonl", eval_pairs)
    write_jsonl_to_s3(bucket, f"{project_name}/processed/corpus.jsonl", chunks)
    logger.info("Embedding data pipeline complete")


# === S3 I/O ===

def read_raw_sources(bucket: str, path: str, sources: list[dict]) -> list[dict]:
    s3 = boto3.client("s3")
    records = []
    for source in sources:
        prefix = f"{path.rstrip('/')}/{source.get('path', '').lstrip('/')}"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                try:
                    resp = s3.get_object(Bucket=bucket, Key=key)
                    content = resp["Body"].read().decode("utf-8", errors="replace").strip()
                    if content:
                        records.append({"source": key, "content": content})
                except Exception as e:
                    logger.warning("Failed to read %s: %s", key, e)
    return records


def write_jsonl_to_s3(bucket: str, key: str, records: list[dict]):
    s3 = boto3.client("s3")
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))


def _write_output_stats(bucket: str, project_name: str, records: list[dict]):
    """Compute output token stats and write to S3 for serving config.

    Estimates token count as chars/4 (rough approximation for English text).
    Stores max, avg, p95 output tokens so the serving stage can set
    max_tokens and reasoning_budget based on actual data.
    """
    outputs = [r.get("output", "") for r in records if r.get("output")]
    if not outputs:
        return
    # Approximate tokens as chars / 4
    token_counts = sorted(len(o) // 4 for o in outputs)
    n = len(token_counts)
    stats = {
        "max_output_tokens": token_counts[-1],
        "avg_output_tokens": round(sum(token_counts) / n),
        "p95_output_tokens": token_counts[int(n * 0.95)],
        "sample_count": n,
    }
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=f"{project_name}/processed/output_stats.json",
        Body=json.dumps(stats).encode("utf-8"))
    logger.info("Output stats: max=%d, avg=%d, p95=%d tokens",
                stats["max_output_tokens"], stats["avg_output_tokens"],
                stats["p95_output_tokens"])


# === Synthetic Generation ===

def generate_synthetic(records, model, endpoint, target_pairs, domain, tools=None, labels=None, output_format="pipe_delimited"):
    backend = _select_backend(model, endpoint)
    concurrency = _calculate_concurrency(model) if not endpoint else 20

    tool_description = tools if isinstance(tools, str) else "\n".join(
        f"- {t.get('name', '')}: {t.get('description', '')}" for t in (tools or [])
    )

    n_batches = (target_pairs + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("Generating %d pairs in %d batches of %d (concurrency=%d, format=%s)",
                target_pairs, n_batches, BATCH_SIZE, concurrency, output_format)

    all_records = []
    failed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for i in range(n_batches):
            batch_size = min(BATCH_SIZE, target_pairs - (i * BATCH_SIZE))
            # Sample different source files per batch for better coverage
            batch_samples = _format_source_samples(records, max_samples=10)
            if output_format == "free_form":
                prompt = FREEFORM_GENERATION_PROMPT.format(
                    domain=domain,
                    tool_description=tool_description or "(not specified)",
                    source_samples=batch_samples or "(no source data)",
                    batch_size=batch_size,
                    output_guidelines=_extract_output_guidelines(domain, labels),
                )
            elif output_format == "scoring":
                prompt = SCORING_GENERATION_PROMPT.format(
                    domain=domain,
                    tool_description=tool_description or "(not specified)",
                    source_samples=batch_samples or "(no source data)",
                    batch_size=batch_size,
                    scoring_guidelines=_extract_scoring_guidelines(domain),
                )
            else:
                prompt = GENERATION_PROMPT.format(
                    domain=domain,
                    tool_description=tool_description or "(not specified)",
                    source_samples=batch_samples or "(no source data)",
                    batch_size=batch_size,
                    valid_labels=_extract_valid_labels(domain, labels),
                )
            future = pool.submit(_call_with_retry, backend, prompt)
            futures[future] = (i, batch_size)

        for future in as_completed(futures):
            batch_idx, expected = futures[future]
            try:
                response = future.result()
                if not response:
                    failed += 1
                    continue
                batch_records = _parse_jsonl(response)
                all_records.extend(batch_records)
                logger.info("Batch %d/%d: %d/%d valid",
                            batch_idx + 1, n_batches, len(batch_records), expected)
                sys.stderr.flush()
            except Exception as e:
                failed += 1
                logger.warning("Batch %d failed: %s", batch_idx, e)

    logger.info("Done: %d valid, %d batches failed", len(all_records), failed)
    return all_records[:target_pairs]


# === Parsing & Helpers ===

def _parse_jsonl(response):
    if not response:
        return []
    response = response.strip()
    if response.startswith("```"):
        parts = response.split("```")
        response = parts[1] if len(parts) >= 2 else response
        if response.startswith("json") or response.startswith("jsonl"):
            response = response.split("\n", 1)[1] if "\n" in response else ""
    records = []
    for line in response.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("output") is not None and obj.get("input"):
                records.append({
                    "instruction": str(obj.get("instruction", "")).strip(),
                    "input": str(obj.get("input", "")).strip(),
                    "output": str(obj.get("output", "")).strip(),
                })
        except json.JSONDecodeError:
            continue
    return records


def _format_source_samples(records, max_samples=10):
    if not records:
        return ""
    samples = random.sample(records, min(max_samples, len(records)))
    return "\n---\n".join(
        f"[{r.get('source', 'unknown')}]\n{r.get('content', '')[:500]}"
        for r in samples
    )


def _check_label_balance(records, labels_config=None, min_per_class=50):
    """Check label distribution and warn about underrepresented classes."""
    from collections import Counter
    dist = Counter()
    for r in records:
        out = r.get("output", "").strip()
        label = out.split("|")[0].strip() if "|" in out else out
        dist[label] = dist.get(label, 0) + 1

    total = len(records)
    logger.info("Label distribution (%d records):", total)
    for label, count in dist.most_common():
        pct = count / total * 100
        marker = " ⚠ LOW" if count < min_per_class else ""
        logger.info("  %s: %d (%.1f%%)%s", label, count, pct, marker)

    if labels_config and isinstance(labels_config, dict):
        first_field = next(iter(labels_config.values()), [])
        if isinstance(first_field, list):
            expected = set(str(v).lower() for v in first_field)
            actual = set(dist.keys())
            missing = expected - actual
            unexpected = actual - expected
            if missing:
                logger.warning("Missing labels (in config but not in data): %s", ", ".join(sorted(missing)))
                logger.warning("Add more raw source data for these intents.")
            if unexpected:
                logger.warning("Unexpected labels (in data but not in config): %s", ", ".join(sorted(unexpected)))

    low = [l for l, c in dist.items() if c < min_per_class]
    if low:
        logger.warning("%d label(s) below %d samples: %s", len(low), min_per_class, ", ".join(low))
        logger.warning("Add more raw source data for underrepresented labels to improve accuracy.")


def _validate_scoring(records, lo=0.0, hi=1.0):
    """Keep only records whose output parses as a float within [lo, hi].

    The model is asked to emit a bare decimal; normalize the stored output to a
    canonical string form so the trainer can parse it deterministically.
    """
    valid = []
    for r in records:
        raw = r.get("output", "").strip()
        # Strip stray characters the model may add (%, quotes, trailing words).
        cleaned = raw.replace("%", "").strip().strip('"').strip()
        # Take the first whitespace-separated token in case of extra prose.
        token = cleaned.split()[0] if cleaned.split() else ""
        try:
            val = float(token)
        except ValueError:
            continue
        # A percent-style value (e.g. "85") gets normalized to [0,1].
        if val > hi and "%" in raw:
            val = val / 100.0
        if lo <= val <= hi:
            r = dict(r)
            r["output"] = f"{val:.4f}"
            valid.append(r)
    return valid


def _check_score_distribution(records):
    """Log score distribution across coarse buckets so we can spot clustering."""
    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    vals = []
    for r in records:
        try:
            v = float(r.get("output", ""))
        except ValueError:
            continue
        vals.append(v)
        if v < 0.2:
            buckets["0.0-0.2"] += 1
        elif v < 0.4:
            buckets["0.2-0.4"] += 1
        elif v < 0.6:
            buckets["0.4-0.6"] += 1
        elif v < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1
    total = len(vals)
    if not total:
        return
    mean = sum(vals) / total
    logger.info("Score distribution (%d records, mean=%.3f):", total, mean)
    for rng, count in buckets.items():
        pct = count / total * 100
        marker = " ⚠ EMPTY" if count == 0 else ""
        logger.info("  %s: %d (%.1f%%)%s", rng, count, pct, marker)
    empty = [rng for rng, c in buckets.items() if c == 0]
    if empty:
        logger.warning("Score buckets with no examples: %s", ", ".join(empty))
        logger.warning("A regression head learns best when scores span the full range.")


def _extract_scoring_guidelines(domain_text):
    """Build scoring guidelines for the scoring generation prompt.

    The domain description carries the rubric (what makes a score high vs low);
    pass it through so the generator anchors the numbers to the task's meaning.
    """
    return (
        "Use the DOMAIN CONTEXT above as the scoring rubric: it defines what a "
        "high score (near 1.0) versus a low score (near 0.0) means for this task. "
        "Assign each input a score consistent with that rubric. Be calibrated — "
        "reserve extreme values for clear cases and use mid-range values for "
        "genuinely ambiguous inputs."
    )


def _extract_valid_labels(domain_text, labels_config=None):
    """Build a valid labels string for the generation prompt.

    If structured labels are provided (from project.labels), use those directly
    and include an explicit output format example showing the pipe-delimited order.
    Otherwise, fall back to extracting underscore_words from domain text.
    """
    if labels_config and isinstance(labels_config, dict):
        parts = []
        field_names = []
        example_values = []
        for field, values in labels_config.items():
            if isinstance(values, list):
                parts.append(f"{field}: {', '.join(str(v) for v in values)}")
                field_names.append(field)
                example_values.append(str(values[0]))
        if parts:
            format_line = " | ".join(f"<{f}>" for f in field_names)
            example_line = "|".join(example_values)
            parts.append(f"\nOutput format: {format_line}")
            parts.append(f"Example output: \"{example_line}\"")
            parts.append("Pick exactly one value from each field, separated by pipe.")
            parts.append("Distribute values across all options in each field — do not heavily favor one value over others.")
            parts.append("For fields representing certainty or confidence, generate examples across the full range: some clear-cut, some ambiguous, some with minimal context.")
            return "\n".join(parts)

    # Fallback: extract from prose
    import re
    labels = re.findall(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b', domain_text.lower())
    seen = set()
    unique = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            unique.append(label)
    if unique:
        return ", ".join(unique)
    return "(no specific labels found in domain description)"


def _extract_output_guidelines(domain_text, labels_config=None):
    """Build output structure guidelines for free-form generation prompt.

    Uses labels as structural categories (e.g., error_type, severity, resource)
    to guide the reasoning output format without enforcing pipe-delimited output.
    """
    if labels_config and isinstance(labels_config, dict):
        parts = ["The response should be structured with the following sections:"]
        for field, values in labels_config.items():
            if isinstance(values, list):
                field_display = field.replace("_", " ").title()
                parts.append(f"- {field_display}: Classify as one of: {', '.join(str(v) for v in values)}")
        parts.append("- Analysis: Explain what is wrong and why it matters")
        parts.append("- Correction: Provide the correct approach or configuration")
        parts.append("- Risk: Describe what happens if the issue is not fixed")
        return "\n".join(parts)

    return (
        "Structure the response with: (1) Issue identification, "
        "(2) Why it's wrong, (3) Correct approach, (4) Risk assessment. "
        "Keep the response factual and actionable."
    )


# === Bedrock / OpenAI Backends ===

def _get_bedrock_rpm_quota(model_id):
    try:
        client = boto3.client("service-quotas")
        paginator = client.get_paginator("list_service_quotas")
        for page in paginator.paginate(ServiceCode="bedrock"):
            for quota in page.get("Quotas", []):
                name = quota.get("QuotaName", "").lower()
                family = model_id.split(".")[-1].split("-")[0]
                if "request" in name and family in name and "per minute" in name:
                    rpm = int(quota.get("Value", 0))
                    if rpm > 0:
                        return rpm
    except Exception as e:
        logger.debug("Could not query Bedrock quota: %s", e)
    return 0


def _calculate_concurrency(model_id):
    rpm = _get_bedrock_rpm_quota(model_id)
    if rpm > 0:
        concurrency = min(20, int(rpm * 0.8 * 3 / 60))
        logger.info("Bedrock quota: %d RPM → concurrency: %d", rpm, concurrency)
        return max(5, concurrency)
    return 20


def _select_backend(model, endpoint):
    if endpoint:
        return OpenAIBackend(model=model, endpoint=endpoint)
    return BedrockBackend(model=model)


def _call_with_retry(backend, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return backend.generate(prompt)
        except Exception as e:
            if "ThrottlingException" in str(e):
                time.sleep(2 ** (attempt + 1))
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error("Failed after %d attempts: %s", max_retries, e)
    return None


class BedrockBackend:
    ALIASES = {
        "claude-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "nova-pro": "amazon.nova-pro-v1:0",
        "nova-lite": "amazon.nova-lite-v1:0",
    }

    def __init__(self, model):
        self.model_id = self.ALIASES.get(model, model)

    def generate(self, prompt):
        from botocore.config import Config
        config = Config(read_timeout=300, connect_timeout=10, retries={"max_attempts": 2})
        client = boto3.client("bedrock-runtime", config=config)
        response = client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 8192, "temperature": 0.8},
        )
        return response["output"]["message"]["content"][0]["text"]


class OpenAIBackend:
    def __init__(self, model, endpoint):
        from openai import OpenAI
        self.client = OpenAI(base_url=endpoint, api_key="not-needed")
        self.model = model

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192, temperature=0.8,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    main()
