import pytest

from app.retry import retry

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.mark.anyio
async def test_retry_succeeds_after_failure(monkeypatch):
    attempts = 0

    async def fake_sleep(_):
        return None

    monkeypatch.setattr("app.retry.asyncio.sleep", fake_sleep)

    @retry(max_attempts=3, retry_on=(ValueError,))
    async def unstable_function():
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise ValueError("Temporary failure")

        return "success"

    result = await unstable_function()

    assert result == "success"
    assert attempts == 3


@pytest.mark.anyio
async def test_retry_raises_after_max_attempts(monkeypatch):
    attempts = 0

    async def fake_sleep(_):
        return None

    monkeypatch.setattr("app.retry.asyncio.sleep", fake_sleep)

    @retry(max_attempts=3, retry_on=(ValueError,))
    async def failing_function():
        nonlocal attempts
        attempts += 1
        raise ValueError("Still failing")

    with pytest.raises(ValueError, match="Still failing"):
        await failing_function()

    assert attempts == 3


@pytest.mark.anyio
async def test_non_retryable_exception_is_not_retried(monkeypatch):
    attempts = 0

    async def fake_sleep(_):
        return None

    monkeypatch.setattr("app.retry.asyncio.sleep", fake_sleep)

    @retry(max_attempts=3, retry_on=(ValueError,))
    async def failing_function():
        nonlocal attempts
        attempts += 1
        raise TypeError("Invalid input")

    with pytest.raises(TypeError, match="Invalid input"):
        await failing_function()

    assert attempts == 1

@pytest.mark.anyio
async def test_async_generator_retries_before_first_chunk():
    attempts = 0

    @retry(
        max_attempts=2,
        initial_delay=0,
        retry_on=(TimeoutError,),
    )
    async def generate():
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise TimeoutError("Temporary timeout")

        yield "Success"

    result = [chunk async for chunk in generate()]

    assert result == ["Success"]
    assert attempts == 2


@pytest.mark.anyio
async def test_async_generator_does_not_retry_after_first_chunk():
    attempts = 0

    @retry(
        max_attempts=3,
        initial_delay=0,
        retry_on=(TimeoutError,),
    )
    async def generate():
        nonlocal attempts
        attempts += 1

        yield "Partial response"
        raise TimeoutError("Stream interrupted")

    result = []

    with pytest.raises(TimeoutError):
        async for chunk in generate():
            result.append(chunk)

    assert result == ["Partial response"]
    assert attempts == 1
