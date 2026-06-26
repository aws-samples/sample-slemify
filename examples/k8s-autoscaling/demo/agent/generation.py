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
