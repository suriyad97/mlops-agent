"""Provider factory — single place to construct/cache LLM providers per model."""
from functools import lru_cache
from typing import Optional

from src.llm.provider import GitHubModelsProvider


@lru_cache(maxsize=8)
def get_provider(model: Optional[str] = None) -> GitHubModelsProvider:
    """Return a cached provider for the given model id (default from settings)."""
    return GitHubModelsProvider(model=model)
