
import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from app.retry import retry

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

REQUEST_TIMEOUT = 30

UsageCallback = Callable[[dict[str, int | str]], None]


def should_retry(exc: Exception) -> bool:
    if isinstance(
        exc,
        (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException),
    ):
        return True

    if isinstance(exc, errors.APIError):
        return exc.code == 429 or (
            exc.code is not None and 500 <= exc.code <= 599
        )

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or (
            500 <= exc.response.status_code <= 599
        )

    return False


async def generate(
    prompt: str,
    provider: str,
    on_usage: UsageCallback | None = None,
) -> AsyncIterator[str]:
    provider = provider.lower().strip()

    if provider == "gemini":
        if on_usage is None:
            async for chunk in generate_with_gemini(prompt):
                yield chunk
        else:
            async for chunk in generate_with_gemini(prompt, on_usage):
                yield chunk

    elif provider == "ollama":
        if on_usage is None:
            async for chunk in generate_with_ollama(prompt):
                yield chunk
        else:
            async for chunk in generate_with_ollama(prompt, on_usage):
                yield chunk

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


@retry(
    max_attempts=3,
    initial_delay=1,
    backoff_factor=2,
    max_delay=10,
    retry_on=(
        TimeoutError,
        httpx.TimeoutException,
        httpx.HTTPStatusError,
        errors.APIError,
    ),
    retry_if=should_retry,
)
async def generate_with_gemini(
    prompt: str,
    on_usage: UsageCallback | None = None,
) -> AsyncIterator[str]:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT * 1000,
        ),
    )

    async with asyncio.timeout(REQUEST_TIMEOUT):
        stream = await client.aio.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        async for chunk in stream:
            if chunk.text:
                yield chunk.text

            metadata = getattr(chunk, "usage_metadata", None)

            if metadata and on_usage:
                on_usage({
                    "model": getattr(chunk, "model_version", None)
                    or GEMINI_MODEL,
                    "input_tokens": metadata.prompt_token_count or 0,
                    "output_tokens": metadata.candidates_token_count or 0,
                })


@retry(
    max_attempts=3,
    initial_delay=1,
    backoff_factor=2,
    max_delay=10,
    retry_on=(
        TimeoutError,
        httpx.TimeoutException,
        httpx.HTTPStatusError,
    ),
    retry_if=should_retry,
)
async def generate_with_ollama(
    prompt: str,
    on_usage: UsageCallback | None = None,
) -> AsyncIterator[str]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "think": False
    }

    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", url, json=payload
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                data = json.loads(line)

                if data.get("response"):
                    yield data["response"]

                if data.get("done") and on_usage:
                    on_usage({
                        "model": data.get("model", OLLAMA_MODEL),
                        "input_tokens": data.get(
                            "prompt_eval_count", 0
                        ),
                        "output_tokens": data.get(
                            "eval_count", 0
                        ),
                    })