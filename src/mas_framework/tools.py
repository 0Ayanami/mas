from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Any, Callable, Dict, List, Optional

from camel.toolkits import FunctionTool, SearchToolkit, FileWriteToolkit


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, FunctionTool] = {}

    def register_function(
        self,
        fn: Callable[..., Any],
        name: Optional[str] = None,
    ) -> FunctionTool:
        tool = FunctionTool(fn)
        tool_name = name or tool.get_function_name()

        if tool_name in self._tools:
            raise ValueError(f"Tool \"{tool_name}\" already exists in registry.")

        self._tools[tool_name] = tool
        return tool

    def register_tool(
        self,
        tool: FunctionTool,
        name: Optional[str] = None,
    ) -> FunctionTool:
        tool_name = name or tool.get_function_name()

        if tool_name in self._tools:
            raise ValueError(f"Tool \"{tool_name}\" already exists in registry.")

        self._tools[tool_name] = tool
        return tool

    def register_toolkit(
        self,
        toolkit: Any,
        only: Optional[List[str]] = None,
    ) -> None:
        for tool in toolkit.get_tools():
            tool_name = tool.get_function_name()
            if only is not None and tool_name not in only:
                continue
            self.register_tool(tool)

    def get_tool(self, name: str) -> FunctionTool:
        if name not in self._tools:
            raise KeyError(f"Tool \"{name}\" not found.")
        return self._tools[name]

    def get_tools(self) -> List[FunctionTool]:
        return list(self._tools.values())

    def names(self) -> List[str]:
        return list(self._tools.keys())


def build_default_tool_registry(functions: List[Callable[..., Any]]) -> ToolRegistry:
    registry = ToolRegistry()

    # 自定义函数工具
    for f in functions:
        registry.register_function(f)

    # CAMEL built-in toolkits
    registry.register_toolkit(SearchToolkit())
    registry.register_toolkit(FileWriteToolkit(working_directory="./outputs"))

    return registry
