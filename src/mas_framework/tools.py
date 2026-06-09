from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mas_framework.models import ToolCall


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
            known = ", ".join(sorted(self._tools))
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
    return registry
