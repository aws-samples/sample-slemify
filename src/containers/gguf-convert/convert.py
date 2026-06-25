# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CPU-only generation pipeline: download a HuggingFace base model, convert it to
a GGUF (f16), optionally quantize it with llama.cpp, and upload the result to S3.

This replaces the GPU fine-tuning path for generative experts. Slemify serves
generative models stock (no fine-tuning); knowledge comes from RAG at serving
time, not from training. See pkg/config/schema.go for the rationale.

All inputs come from environment variables and are validated before use. External
binaries are invoked with argument lists (never shell strings) so untrusted-looking
values cannot inject shell commands.
"""

import os
import re
import subprocess
import sys

import boto3
from huggingface_hub import snapshot_download

# Quantization types accepted by the config schema. f16/none mean "no quantize"
# (serve the f16 GGUF directly); the rest map to a llama-quantize preset.
QUANT_PRESETS = {
    "q4_k_m": "Q4_K_M",
    "q5_k_m": "Q5_K_M",
    "q8_0": "Q8_0",
}
NO_QUANTIZE = {"f16", "none"}

# Conservative allowlists. These are defense-in-depth: the Go validator already
# constrains project name and bucket, but this container must not trust its env.
SAFE_PROJECT = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
SAFE_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$")
SAFE_MODEL = re.compile(r"^[A-Za-z0-9._\-/]+$")
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._\-]+\.gguf$")

WORK_DIR = "/tmp/convert"
MODEL_DIR = os.path.join(WORK_DIR, "base-model")
F16_PATH = os.path.join(WORK_DIR, "model-f16.gguf")


def require_env(name):
    """Read a required environment variable or exit with a clear message."""
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"ERROR: required environment variable {name} is not set")
    return value


def validate(name, value, pattern):
    """Reject values that don't match the expected safe pattern."""
    if not pattern.match(value):
        sys.exit(f"ERROR: {name} value {value!r} failed validation")
    return value


def log(message):
    print(f"[gguf-convert] {message}", flush=True)


def run(args):
    """Run an external command from an argument list (no shell)."""
    log("running: " + " ".join(args))
    subprocess.run(args, check=True)


def main():
    base_model = validate("BASE_MODEL", require_env("BASE_MODEL"), SAFE_MODEL)
    quantize = require_env("QUANTIZE").lower()
    s3_bucket = validate("S3_BUCKET", require_env("S3_BUCKET"), SAFE_BUCKET)
    project = validate("PROJECT", require_env("PROJECT"), SAFE_PROJECT)
    gguf_filename = validate("GGUF_FILENAME", require_env("GGUF_FILENAME"), SAFE_FILENAME)

    if quantize not in QUANT_PRESETS and quantize not in NO_QUANTIZE:
        sys.exit(
            f"ERROR: QUANTIZE value {quantize!r} is not supported "
            f"(expected one of: {', '.join(sorted(set(QUANT_PRESETS) | NO_QUANTIZE))})"
        )

    os.makedirs(WORK_DIR, exist_ok=True)

    log(f"step 1/4: downloading base model {base_model} from HuggingFace")
    snapshot_download(
        repo_id=base_model,
        local_dir=MODEL_DIR,
        # Weights only: skip the GGUF/other-format files some repos ship so the
        # converter reads the canonical safetensors/PyTorch weights.
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "*.txt", "tokenizer*"],
    )

    log("step 2/4: converting to f16 GGUF")
    run([
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_hf_to_gguf.py"),
        MODEL_DIR,
        "--outfile", F16_PATH,
        "--outtype", "f16",
    ])

    if quantize in NO_QUANTIZE:
        log(f"step 3/4: quantization skipped (QUANTIZE={quantize}); serving f16")
        gguf_path = F16_PATH
    else:
        preset = QUANT_PRESETS[quantize]
        gguf_path = os.path.join(WORK_DIR, gguf_filename)
        log(f"step 3/4: quantizing f16 GGUF to {preset}")
        run(["llama-quantize", F16_PATH, gguf_path, preset])

    size_mb = os.path.getsize(gguf_path) / 1048576
    s3_key = f"models/{project}/{gguf_filename}"
    log(f"step 4/4: uploading {gguf_path} ({size_mb:.1f} MB) to s3://{s3_bucket}/{s3_key}")
    boto3.client("s3").upload_file(gguf_path, s3_bucket, s3_key)

    log(f"done: s3://{s3_bucket}/{s3_key}")


if __name__ == "__main__":
    main()
