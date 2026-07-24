from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def message_count(events: list[Any]) -> int:
    return sum(
        1
        for event in events
        if getattr(event, "source", None)
        and isinstance(getattr(event, "content", None), str)
    )


def token_usage(
    *,
    events: list[Any],
    trace: str,
    proposals: list[Any] | None = None,
) -> dict[str, Any]:
    prompt_tokens, completion_tokens = usage_tokens(events)
    if prompt_tokens is None and completion_tokens is None:
        total = estimate_tokens(trace)
        prompt_tokens = None
        completion_tokens = None
        source = "estimated"
    else:
        total = (prompt_tokens or 0) + (completion_tokens or 0)
        source = "api_usage"
    proposal_tokens = sum(
        estimate_tokens(
            json.dumps(
                proposal.to_dict() if hasattr(proposal, "to_dict") else proposal,
                ensure_ascii=False,
                default=str,
            )
        )
        for proposal in (proposals or [])
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
        "token_source": source,
        "memory_proposal_tokens": proposal_tokens,
        "memory_proposal_token_source": "estimated",
    }


def usage_tokens(events: list[Any]) -> tuple[int | None, int | None]:
    prompt_total = 0
    completion_total = 0
    found = False
    for event in events:
        for usage in walk_usage(event):
            prompt_value = usage_value(usage, "prompt_tokens", "input_tokens")
            completion_value = usage_value(usage, "completion_tokens", "output_tokens")
            if prompt_value is not None:
                prompt_total += prompt_value
                found = True
            if completion_value is not None:
                completion_total += completion_value
                found = True
    if not found:
        return None, None
    return prompt_total, completion_total


def walk_usage(value: Any) -> Iterable[Any]:
    usage = getattr(value, "models_usage", None) or getattr(value, "usage", None)
    if usage is not None:
        yield usage
    messages = getattr(value, "messages", None)
    if messages:
        for message in messages:
            yield from walk_usage(message)


def usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        if isinstance(usage, dict) and usage.get(name) is not None:
            return int(usage[name])
        if getattr(usage, name, None) is not None:
            return int(getattr(usage, name))
    return None


def estimate_tokens(text: str) -> int:
    return max(1, len(str(text)) // 4)


def final_weight_snapshots(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    for decision in reversed(decisions):
        update = decision.get("metadata", {}).get("weight_snapshots_after_update")
        if update:
            return update
    return {}


def sum_metric(results: Iterable[Any], key: str) -> float:
    total = 0.0
    for result in results:
        metrics = getattr(result, "run_metrics", None) or getattr(result, "metrics", {})
        total += float(metrics.get(key, 0) or 0)
    return total


__all__ = [
    "estimate_tokens",
    "final_weight_snapshots",
    "message_count",
    "sum_metric",
    "token_usage",
    "usage_tokens",
]
