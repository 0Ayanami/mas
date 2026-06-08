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


def read_text_file(path: str, max_chars: int = 12000) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    return file_path.read_text(encoding="utf-8")[:max_chars]


def keyword_extract(text: str, keywords: str) -> str:
    wanted = [item.strip().lower() for item in keywords.split(",") if item.strip()]
    lines = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(keyword in lowered for keyword in wanted):
            lines.append(line)
    return "\n".join(lines[:80]) or "No matching lines found."


def build_default_tool_registry(memory_search: Callable[[str, int], str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="read_text_file",
            description="Read a UTF-8 text or markdown file from disk.",
            func=read_text_file,
        )
    )
    registry.register(
        ToolSpec(
            name="keyword_extract",
            description="Extract lines containing comma-separated keywords from a text block.",
            func=keyword_extract,
        )
    )
    if memory_search is not None:
        registry.register(
            ToolSpec(
                name="search_memory",
                description="Search accepted and rejected shared memory proposals.",
                func=memory_search,
            )
        )
    return registry
