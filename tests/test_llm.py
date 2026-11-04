from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm import generate_with_gemini


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def mock_async_stream(chunks):
    for chunk in chunks:
        yield chunk


@pytest.mark.anyio
async def test_gemini_streams_text_chunks():
    chunk1 = MagicMock()
    chunk1.text = "Hello "

    chunk2 = MagicMock()
    chunk2.text = "World!"

    mock_stream = mock_async_stream([chunk1, chunk2])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        return_value=mock_stream
    )

    with (
        patch("app.llm.genai.Client", return_value=mock_client),
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-api-key"}),
    ):
        result = [
            chunk async for chunk in generate_with_gemini("Say hello")
        ]

    assert result == ["Hello ", "World!"]

    mock_client.aio.models.generate_content_stream.assert_awaited_once()