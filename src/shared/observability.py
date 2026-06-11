"""Arize Phoenix observability via OpenInference / OpenTelemetry.

Registers a Phoenix tracer provider and auto-instruments the OpenAI client
(every LLM call: prompts, completions, token counts, latency) and
langchain-core (every LangGraph node/subgraph invocation), giving full
hierarchical traces of supervisor -> agent -> node -> LLM call.

Degrades to a no-op when Phoenix packages are missing or the collector is
disabled, so the platform never fails because observability is unavailable.
"""
from typing import Optional

from src.config.settings import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)
_initialized = False


def setup_observability() -> bool:
    """Idempotent. Returns True when tracing is active."""
    global _initialized
    if _initialized:
        return True

    settings = get_settings()
    if not settings.phoenix_enabled:
        logger.info("phoenix observability disabled via settings")
        return False

    try:
        from phoenix.otel import register
    except ImportError:
        logger.warning("arize-phoenix-otel not installed — tracing disabled")
        return False

    try:
        endpoint = f"{settings.phoenix_collector_endpoint.rstrip('/')}/v1/traces"
        tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=endpoint,
            batch=True,
        )
        _instrument(tracer_provider)
        _initialized = True
        logger.info(
            "phoenix tracing active: project=%s endpoint=%s",
            settings.phoenix_project_name, endpoint,
        )
        return True
    except Exception as exc:
        logger.warning("phoenix setup failed — tracing disabled: %s", exc)
        return False


def _instrument(tracer_provider) -> None:
    """Attach OpenInference instrumentors for the libraries we actually use."""
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    except ImportError:
        logger.warning("openinference-instrumentation-openai not installed — LLM spans missing")

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    except ImportError:
        logger.warning("openinference-instrumentation-langchain not installed — graph spans missing")


def get_tracer(name: str) -> Optional[object]:
    """OTel tracer for custom spans inside nodes/tools (None when inactive)."""
    if not _initialized:
        return None
    from opentelemetry import trace
    return trace.get_tracer(name)
