from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, Mapping, TypeVar

from dotenv import load_dotenv

T = TypeVar("T")

def _default_model_info() -> dict[str, Any]:
    return {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    }

def run_autogen_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run an AutoGen coroutine from synchronous framework code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def build_model_client(
    model: str | None = None,
    temperature: float | None = None,
    model_config: Mapping[str, Any] | None = None,
    model_info: Mapping[str, Any] | None = None,
) -> Any:
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("No model API key configured. Set API_KEY in .env or the environment.")

    selected_model = model or os.getenv("MODEL")
    if not selected_model:
        raise ValueError("No model configured. Pass model from experiment config or set MODEL.")

    kwargs = dict(model_config or {})
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    kwargs.setdefault("parallel_tool_calls", False)
    kwargs["api_key"] = api_key
    kwargs["model"] = selected_model
    base_url = os.getenv("BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    if model_info is not None:
        kwargs["model_info"] = dict(model_info)

    try:
        return OpenAIChatCompletionClient(**kwargs)
    except ValueError as exc:
        if "model_info" not in str(exc):
            raise
        kwargs["model_info"] = _default_model_info()
        return OpenAIChatCompletionClient(**kwargs)

def _normalize_agent_name(agent_name: str) -> str:
    return agent_name.lower().replace(" ", "_")


def _base_agent_name(agent_name: str) -> str:
    for suffix in (" A", " B"):
        if agent_name.endswith(suffix):
            return agent_name[: -len(suffix)]
    return agent_name

def _agent_outputs(events: list[Any]) -> list[tuple[str, str]]:
    outputs: list[tuple[str, str]] = []
    for event in events:
        source = getattr(event, "source", None)
        content = getattr(event, "content", None)
        if source and source != "user" and isinstance(content, str) and content.strip():
            outputs.append((source, content.strip()))
    return outputs


def _final_text(events: list[Any]) -> str:
    for event in reversed(events):
        source = getattr(event, "source", None)
        content = getattr(event, "content", None)
        if source and source != "user" and isinstance(content, str) and content.strip():
            return content
        messages = getattr(event, "messages", None)
        if messages:
            for message in reversed(messages):
                msg_source = getattr(message, "source", None)
                msg_content = getattr(message, "content", None)
                if msg_source != "user" and isinstance(msg_content, str) and msg_content.strip():
                    return msg_content
    return ""

__all__ = [
    "_agent_outputs",
    "_base_agent_name",
    "_final_text",
    "_normalize_agent_name",
    "build_model_client",
    "run_autogen_sync",
]
