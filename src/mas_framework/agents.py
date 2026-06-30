from __future__ import annotations

import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, Protocol, TypeVar

from mas_framework.memory import Mem0MemoryBackend
from mas_framework.models import AgentConfig, AgentState
from mas_framework.tools import ToolRegistry, build_default_tool_registry

T = TypeVar("T")


class AgentProtocol(Protocol):
    config: AgentConfig
    state: AgentState
    memory: Mem0MemoryBackend

    def run(self, prompt: str) -> str:
        """Run the agent and return its final text response."""
        ...

    def run_stream(self, prompt: str) -> list[Any]:
        """Run the agent through AutoGen streaming and return emitted events."""
        ...


def _run_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run an AutoGen coroutine from synchronous framework code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def _safe_agent_name(agent_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", agent_id)
    if not name or name[0].isdigit():
        name = f"agent_{name}"
    return name


def _default_model_info() -> dict[str, Any]:
    return {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    }


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        memory: Mem0MemoryBackend | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.config = config
        self.memory = memory or Mem0MemoryBackend()
        self.tools = tools or build_default_tool_registry(memory_backend=self.memory)
        self.state = AgentState()
        self._model_client = config.model_client or self._build_model_client()
        self.last_stream_events: list[Any] = []
        self._agent = self._build_agent()

    def _build_model_client(self) -> Any:
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        api_key = (
            self.config.api_key
            or os.getenv("API_KEY")
        )
        if not api_key:
            raise ValueError(
                "No model API key configured. Set API_KEY or AgentConfig.api_key."
            )

        kwargs = dict(self.config.model_config_dict)
        kwargs["api_key"] = api_key
        kwargs["model"] = (self.config.model or os.getenv("MODEL"))
        kwargs["base_url"] = (self.config.base_url or os.getenv("BASE_URL"))
        kwargs["model_info"] = self.config.model_info

        try:
            return OpenAIChatCompletionClient(**kwargs)
        except ValueError as exc:
            if "model_info" not in str(exc):
                raise
            kwargs["model_info"] = _default_model_info()
            return OpenAIChatCompletionClient(**kwargs)

    def _build_agent(self) -> Any:
        from autogen_agentchat.agents import AssistantAgent

        tools = self.tools.get_tools()
        return AssistantAgent(
            name=_safe_agent_name(self.config.agent_id),
            description=self.config.role or self.config.agent_id,
            model_client=self._model_client,
            system_message=self.config.system_prompt,
            tools=tools or None,
            reflect_on_tool_use=self.config.reflect_on_tool_use,
            max_tool_iterations=self.config.max_tool_iterations,
        )

    @staticmethod
    def _response_text(result: Any) -> str:
        for message in reversed(result.messages):
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if hasattr(content, "model_dump_json"):
                return content.model_dump_json()
            if content is not None:
                return json.dumps(content, ensure_ascii=False, default=str)
        raise ValueError("AutoGen agent returned no text response.")

    async def _collect_stream(self, prompt: str) -> list[Any]:
        events = []
        async for event in self._agent.run_stream(task=prompt):
            events.append(event)
        return events

    @staticmethod
    def _stream_task_result(events: list[Any]) -> Any:
        for event in reversed(events):
            if hasattr(event, "messages") and hasattr(event, "stop_reason"):
                return event
        raise ValueError("AutoGen run_stream returned no TaskResult.")

    def run(self, prompt: str) -> str:
        events = _run_sync(self._collect_stream(prompt))
        self.last_stream_events = events
        result = self._stream_task_result(events)
        return self._response_text(result)

    def run_stream(self, prompt: str) -> list[Any]:
        events = _run_sync(self._collect_stream(prompt))
        self.last_stream_events = events
        return events

    def reset(self) -> None:
        """Reset AutoGen conversation context while retaining state and memory."""
        _run_sync(self._agent.reset())

    def close(self) -> None:
        """Close the underlying AutoGen model client."""
        close = getattr(self._model_client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                _run_sync(result)


def create_agent(
    config: AgentConfig,
    memory: Mem0MemoryBackend | None = None,
    tools: ToolRegistry | None = None,
) -> AgentProtocol | None:
    """Create an AutoGen-backed agent while preserving the existing factory contract."""
    has_key = bool(
        config.api_key
        or os.getenv("API_KEY")
    )
    if config.model_client is None and not has_key:
        print(f"Warning: No API key found for {config.agent_id}. Agent will not be backed by LLM.")
        return None
    try:
        return Agent(config, memory=memory, tools=tools)
    except Exception as exc:
        print(f"Agent init failed for {config.agent_id}; Error: {exc}")
        return None


__all__ = ["Agent", "AgentProtocol", "create_agent"]
