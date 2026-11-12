import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    retry_if: Callable[[Exception], bool] | None = None,
):
    def decorator(func: Callable[..., Any]):

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_generator_wrapper(*args, **kwargs):
                delay = initial_delay

                for attempt in range(1, max_attempts + 1):
                    emitted = False

                    try:
                        async for item in func(*args, **kwargs):
                            emitted = True
                            yield item
                        return

                    except retry_on as exc:
                        # Never restart after sending the first chunk.
                        if emitted:
                            raise

                        if retry_if is not None and not retry_if(exc):
                            raise

                        if attempt == max_attempts:
                            raise

                        logger.warning(
                            "%s failed on attempt %s/%s: %s. Retrying in %.1f seconds.",
                            func.__name__,
                            attempt,
                            max_attempts,
                            exc,
                            delay,
                        )

                        await asyncio.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)

            return async_generator_wrapper

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except retry_on as exc:
                    if retry_if is not None and not retry_if(exc):
                        raise

                    if attempt == max_attempts:
                        raise

                    logger.warning(
                        "%s failed on attempt %s/%s: %s. Retrying in %.1f seconds.",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )

                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

        return async_wrapper

    return decorator
