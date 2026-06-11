"""Minimal ReAct executor: Thought -> Action (tool call) -> Observation loop.

Tools are plain callables registered with a JSON-schema description; the loop
uses native OpenAI tool calling, so the model's 'Thought' is its assistant
content and 'Action' is a tool_call. Observations are tool results fed back.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from src.llm.provider import GitHubModelsProvider
from src.shared.exceptions import LLMError
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReactTool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def run_react(
    provider: GitHubModelsProvider,
    system: str,
    user: str,
    tools: List[ReactTool],
    max_steps: int = 8,
) -> str:
    """Run a ReAct loop and return the model's final answer text."""
    registry = {t.name: t for t in tools}
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for step in range(max_steps):
        response = provider.client.chat.completions.create(
            model=provider.model,
            temperature=0.1,
            messages=messages,
            tools=[t.to_openai() for t in tools],
        )
        message = response.choices[0].message
        if not message.tool_calls:
            if not message.content:
                raise LLMError("ReAct loop produced neither tool calls nor an answer")
            return message.content

        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            tool = registry.get(call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
                observation = str(tool.fn(**args)) if tool else f"unknown tool {call.function.name}"
            except Exception as exc:  # observation, not crash — the model should react to it
                observation = f"tool error: {exc}"
            logger.info("react step=%d tool=%s", step, call.function.name)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": observation[:8000],
            })
    raise LLMError(f"ReAct loop exceeded {max_steps} steps without a final answer")
