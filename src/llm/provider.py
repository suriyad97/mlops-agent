"""GitHub Models provider with plain, structured, and tool-calling completions."""
import json
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.config.settings import get_settings
from src.shared.exceptions import LLMError
from src.shared.logging import get_logger
from src.shared.retry import retry

TModel = TypeVar("TModel", bound=BaseModel)
logger = get_logger(__name__)

GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"


class GitHubModelsProvider:
    """Thin, stateless wrapper around the GitHub Models OpenAI-compatible endpoint."""

    def __init__(self, model: Optional[str] = None) -> None:
        settings = get_settings()
        self.model = model or settings.github_model
        self._client = OpenAI(base_url=GITHUB_MODELS_BASE_URL, api_key=settings.github_token)

    @property
    def client(self) -> OpenAI:
        return self._client

    @retry(times=3, delay=2.0, exceptions=(LLMError,))
    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Plain text completion."""
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMError("empty completion")
        return content

    # GitHub Models free tier caps gpt-4o requests at 8000 input tokens.
    # ~4 chars/token; keep prompts well under the ceiling.
    MAX_USER_CHARS = 18_000

    def complete_structured(
        self,
        schema: Type[TModel],
        system: str,
        user: str,
        max_repair_rounds: int = 2,
    ) -> TModel:
        """JSON-mode completion validated against a Pydantic schema, with self-repair.

        Repair rounds REBUILD the request (system + user + short error note)
        rather than appending the invalid output — appending can exceed the
        provider's request-size limit when the model produces degenerate output.
        """
        schema_json = json.dumps(schema.model_json_schema())
        system_full = (
            f"{system}\n\nRespond ONLY with valid JSON matching this JSON Schema:\n{schema_json}"
        )
        user = user[: self.MAX_USER_CHARS]
        repair_note = ""
        last_error: Optional[Exception] = None
        for round_num in range(max_repair_rounds + 1):
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_full},
                    {"role": "user", "content": user + repair_note},
                ],
            )
            text = response.choices[0].message.content or ""
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                last_error = exc
                logger.warning("structured output invalid (round %d): %s", round_num, str(exc)[:300])
                repair_note = (
                    f"\n\nIMPORTANT: your previous response was invalid JSON "
                    f"({str(exc)[:200]}). Respond with COMPLETE, valid, concise JSON only — "
                    f"no truncation, no runaway escape sequences."
                )
        raise LLMError(f"structured output failed after repair rounds: {last_error}")
