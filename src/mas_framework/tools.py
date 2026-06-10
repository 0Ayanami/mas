from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
from typing import Optional, List, Dict
from mas_framework.models import ToolCall
from mas_framework.utils.memory_proposal_tool import create_proposal_creation_toolkit

ToolFunc = Callable[..., str]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    func: ToolFunc


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.history: list[ToolCall] = []

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def call(self, name: str, **kwargs: Any) -> str:
        if name not in self._tools:
            known = "", "".join(sorted(self._tools))
            raise KeyError(f"Unknown tool '{name}'. Known tools: {known}")
        result = self._tools[name].func(**kwargs)
        self.history.append(ToolCall(name=name, arguments=kwargs, result=result))
        return result

    def camel_tools(self) -> list[ToolFunc]:
        return [spec.func for spec in self._tools.values()]

    def describe(self) -> str:
        return "\n".join(f"- {spec.name}: {spec.description}" for spec in self._tools.values())


def build_default_tool_registry(memory_search: Callable[[str, int], str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    if memory_search is not None:
        registry.register(
            ToolSpec(
                name="search_memory",
                description="Search accepted shared memory proposals.",
                func=memory_search,
            )
        )
    
    toolkit = create_proposal_creation_toolkit()
    
    # 注册提案相关工具
    registry.register(
        ToolSpec(
            name="prepare_proposal_for_submission",
            description=(
                "Prepare a memory proposal for submission to the orchestrator. "
                "This tool creates a complete proposal with header, body, and self-verification "
                "after a ReAct cycle is completed. Parameters: agent_id, task_id, "
                "current_thoughts, current_action, current_results, title, memory_type, "
                "confidence, verification_rationale."
            ),
            func=toolkit["prepare_proposal_for_submission"]
        )
    )
    
    registry.register(
        ToolSpec(
            name="build_memory_proposal",
            description=(
                "Build a memory proposal with thoughts, actions, and observations. "
                "Parameters: agent_id, task_id, thoughts, action_description, result_observation, "
                "title, memory_type, parent_ids, proposal_summary."
            ),
            func=toolkit["build_memory_proposal"]
        )
    )
    
    registry.register(
        ToolSpec(
            name="self_verify_proposal",
            description=(
                "Perform self-verification on a memory proposal. "
                "Parameters: proposal_json, veracity, rationality, value, security, "
                "confidence, rationale."
            ),
            func=toolkit["self_verify_proposal"]
        )
    )
    
    return registry
