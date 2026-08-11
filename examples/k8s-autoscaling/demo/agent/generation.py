"""Answer generation: stream tokens from the auditor SLM (default, on CPU) or
the Bedrock LLM (escalation / fallback). Both take the assembled RAG context.
"""
import asyncio
import json

import httpx

from . import config
from . import prompts


async def stream_slm(text: str, context: str = ""):
    """Stream tokens from the auditor SLM (OpenAI-compatible /v1/chat/completions)."""
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": prompts.auditor_prompt(text, context)}],
        "max_tokens": 1024,
        "temperature": 0.1,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{config.AUDITOR_URL}/v1/chat/completions", json=body) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    content = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


async def propose_fix(prompt: str, json_schema: dict) -> dict | None:
    """Ask the auditor SLM (CPU) for a schema-constrained remediation proposal.
    Passed as response_format.json_schema, llama.cpp's server compiles this into
    a GBNF grammar and masks the sampler, so the model CANNOT emit a field name
    or JSON shape outside json_schema, regardless of what it 'wants' to say.
    Returns the parsed dict, or None if the call/parse failed (caller treats
    that as no_fix, never as a license to guess)."""
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "fix_proposal", "schema": json_schema}},
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{config.AUDITOR_URL}/v1/chat/completions", json=body)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception:
        return None


async def stream_llm(text: str, context: str = ""):
    """Stream tokens from the Bedrock LLM (Converse API)."""
    async for tok in _converse_stream(prompts.llm_prompt(text, context)):
        yield tok


async def stream_calibrated(text: str, context: str = "", reason: str = ""):
    """Stream a calibrated, abstention-aware answer when the gate could not
    confirm the draft (the top-of-ladder LLM answer included). Asserts only what
    the evidence supports and flags what it could not verify."""
    async for tok in _converse_stream(prompts.calibration_prompt(text, context, reason)):
        yield tok


async def _converse_stream(user_content: str):
    resp = config.bedrock.converse_stream(
        modelId=config.LLM_MODEL,
        messages=[{"role": "user", "content": [{"text": user_content}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.2},
    )
    loop = asyncio.get_event_loop()
    stream_iter = iter(resp["stream"])
    while True:
        event = await loop.run_in_executor(None, lambda: next(stream_iter, None))
        if event is None:
            break
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            yield delta["text"]
