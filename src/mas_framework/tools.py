from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from autogen_core.tools import BaseTool, FunctionTool
from typing_extensions import Annotated


ToolCallable = Callable[..., Any]
AutoGenTool = BaseTool[Any, Any]


def _named_callable(fn: ToolCallable, name: str) -> ToolCallable:
    if fn.__name__ == name:
        return fn

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    wrapper.__name__ = name
    return wrapper


def _function_description(fn: ToolCallable, explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    doc = (fn.__doc__ or "").strip()
    return doc or f"Call the `{fn.__name__}` tool."


def make_function_tool(
    fn: ToolCallable,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
) -> FunctionTool:
    """Create an AutoGen FunctionTool with an explicit, model-facing description."""
    tool_name = name or fn.__name__
    return FunctionTool(
        _named_callable(fn, tool_name),
        name=tool_name,
        description=_function_description(fn, description),
        strict=strict,
    )


def make_agent_tool(
    agent: Any,
    *,
    return_value_as_last_message: bool = False,
) -> AutoGenTool:
    """Create an AutoGen AgentTool from a BaseChatAgent."""
    from autogen_agentchat.tools import AgentTool

    return AgentTool(
        agent=agent,
        return_value_as_last_message=return_value_as_last_message,
    )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, AutoGenTool] = {}

    def register_function(
        self,
        fn: ToolCallable,
        name: str | None = None,
        description: str | None = None,
        *,
        strict: bool = False,
    ) -> FunctionTool:
        tool = make_function_tool(
            fn,
            name=name,
            description=description,
            strict=strict,
        )
        self.register_tool(tool)
        return tool

    def register_tool(
        self,
        tool: AutoGenTool | ToolCallable,
        name: str | None = None,
        description: str | None = None,
        *,
        strict: bool = False,
    ) -> AutoGenTool:
        if isinstance(tool, BaseTool):
            if name is not None or description is not None or strict:
                raise ValueError("Name/description overrides are only supported for callables.")
            registered = tool
        else:
            registered = make_function_tool(
                tool,
                name=name,
                description=description,
                strict=strict,
            )

        if registered.name in self._tools:
            raise ValueError(f'Tool "{registered.name}" already exists in registry.')
        self._tools[registered.name] = registered
        return registered

    def register_agent_tool(
        self,
        agent: Any,
        *,
        return_value_as_last_message: bool = False,
    ) -> AutoGenTool:
        tool = make_agent_tool(
            agent,
            return_value_as_last_message=return_value_as_last_message,
        )
        self.register_tool(tool)
        return tool

    def get_tool(self, name: str) -> AutoGenTool:
        if name not in self._tools:
            raise KeyError(f'Tool "{name}" not found.')
        return self._tools[name]

    def get_tools(self) -> list[AutoGenTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools.values())


def read_text_file(
    path: Annotated[str, "Path to a UTF-8 text file to read."],
    max_chars: Annotated[int, "Maximum number of characters to return."] = 20_000,
) -> str:
    """Read a UTF-8 text file and return at most max_chars characters."""
    return Path(path).read_text(encoding="utf-8")[:max_chars]


def write_text_file(
    path: Annotated[str, "Relative path under the local outputs directory."],
    content: Annotated[str, "UTF-8 text content to write."],
) -> str:
    """Write UTF-8 text under the local outputs directory and return its path."""
    output_root = (Path.cwd() / "outputs").resolve()
    target = (output_root / path).resolve()
    if target != output_root and output_root not in target.parents:
        raise ValueError("File writes must stay inside the outputs directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target)


def web_search(
    query: Annotated[str, "Search query keywords."],
    max_results: Annotated[int, "Maximum number of compact results to return."] = 5,
) -> str:
    """Search the web through DuckDuckGo Instant Answer and return compact JSON."""
    params = urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
    )
    request = Request(
        f"https://api.duckduckgo.com/?{params}",
        headers={"User-Agent": "consensus-memory-mas/0.1"},
    )
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results: list[dict[str, str]] = []
    if payload.get("AbstractText"):
        results.append(
            {
                "title": payload.get("Heading", ""),
                "snippet": payload["AbstractText"],
                "url": payload.get("AbstractURL", ""),
            }
        )
    for topic in payload.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        candidates = topic.get("Topics", [topic])
        for candidate in candidates:
            if len(results) >= max_results:
                break
            if candidate.get("Text"):
                results.append(
                    {
                        "title": candidate["Text"].split(" - ", 1)[0],
                        "snippet": candidate["Text"],
                        "url": candidate.get("FirstURL", ""),
                    }
                )
    return json.dumps(results, ensure_ascii=False)


def search_shared_memory(
    query: Annotated[str, "Natural-language query for accepted MAS shared memories."],
    user_id: Annotated[str | None, "Optional mem0 user id or agent namespace."] = None,
    *,
    memory_backend: Any,
) -> str:
    """Search accepted shared memories stored in mem0 and return JSON results."""
    results = memory_backend.search(query=query, user_id=user_id)
    return json.dumps(results, ensure_ascii=False, default=str)


def build_duckduckgo_http_tool() -> AutoGenTool:
    """Build AutoGen's HttpTool for DuckDuckGo when the optional extra is installed."""
    from autogen_ext.tools.http import HttpTool

    return HttpTool(
        name="web_search",
        description=(
            "Search public web information through DuckDuckGo Instant Answer. "
            "Use this for lightweight fact lookup; results are returned as JSON."
        ),
        scheme="https",
        host="api.duckduckgo.com",
        port=443,
        path="/",
        method="GET",
        return_type="json",
        json_schema={
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "Search query keywords.",
                },
                "format": {
                    "type": "string",
                    "description": "Response format; keep this as json.",
                    "default": "json",
                },
                "no_html": {
                    "type": "integer",
                    "description": "Set to 1 to strip HTML from snippets.",
                    "default": 1,
                },
                "skip_disambig": {
                    "type": "integer",
                    "description": "Set to 1 to skip disambiguation-only results.",
                    "default": 1,
                },
            },
            "required": ["q"],
        },
    )


def build_default_tool_registry(
    functions: Sequence[ToolCallable | AutoGenTool] | None = None,
    *,
    memory_backend: Any | None = None,
    prefer_autogen_http_tool: bool = True,
) -> ToolRegistry:
    registry = ToolRegistry()
    for function in functions or []:
        registry.register_tool(function)

    if memory_backend is not None:
        def search_shared_memory_tool(
            query: Annotated[str, "Natural-language query for accepted MAS shared memories."],
            user_id: Annotated[str | None, "Optional mem0 user id or agent namespace."] = None,
        ) -> str:
            return search_shared_memory(
                query,
                user_id,
                memory_backend=memory_backend,
            )

        registry.register_function(
            search_shared_memory_tool,
            name="search_shared_memory",
            description=(
                "Search accepted consensus memories stored in mem0. "
                "Use this before proposing duplicate or conflicting memory."
            ),
        )

    if prefer_autogen_http_tool:
        try:
            registry.register_tool(build_duckduckgo_http_tool())
        except ModuleNotFoundError:
            registry.register_function(
                web_search,
                description=(
                    "Search public web information through DuckDuckGo Instant Answer "
                    "and return compact JSON results."
                ),
            )
    else:
        registry.register_function(
            web_search,
            description=(
                "Search public web information through DuckDuckGo Instant Answer "
                "and return compact JSON results."
            ),
        )

    registry.register_function(
        read_text_file,
        description="Read a UTF-8 text file from the local filesystem.",
    )
    registry.register_function(
        write_text_file,
        description="Write UTF-8 text under the workspace outputs directory.",
    )
    return registry


__all__ = [
    "AutoGenTool",
    "ToolCallable",
    "ToolRegistry",
    "build_default_tool_registry",
    "build_duckduckgo_http_tool",
    "make_agent_tool",
    "make_function_tool",
    "read_text_file",
    "search_shared_memory",
    "web_search",
    "write_text_file",
]
