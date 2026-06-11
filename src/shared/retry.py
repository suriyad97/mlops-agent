"""Retry decorator with exponential backoff."""
import functools
import time
from typing import Callable, Tuple, Type, TypeVar

from src.shared.logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


def retry(
    times: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            wait = delay
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == times:
                        raise
                    logger.warning(
                        "retrying %s after %s (attempt %d/%d)",
                        fn.__name__, exc, attempt, times,
                    )
                    time.sleep(wait)
                    wait *= backoff
            raise RuntimeError("unreachable")
        return wrapper
    return decorator
