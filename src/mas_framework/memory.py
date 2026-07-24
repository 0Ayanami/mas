from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskMemoryBackend:
    """In-process shared memory for one task, cleared by constructing a new instance."""

    task_id: str = "task"
    accepted_proposals: list[Any] = field(default_factory=list)

    def add_proposal(self, proposal: Any) -> Any:
        consensus_result = _proposal_consensus_result(proposal)
        if consensus_result != "pass":
            raise ValueError(
                "Only memory proposals with consensus_result='pass' can enter task memory."
            )
        self.accepted_proposals.append(proposal)
        return proposal

    def payloads(self) -> list[dict[str, Any]]:
        return [_proposal_payload(proposal) for proposal in self.accepted_proposals]

    def search(self, query: str) -> list[dict[str, Any]]:
        needle = query.casefold()
        results = []
        for payload in self.payloads():
            text = json.dumps(payload, ensure_ascii=False, default=str).casefold()
            if needle in text:
                results.append(payload)
        return results

    def context_text(self) -> str:
        payloads = self.payloads()
        if not payloads:
            return ""
        return (
            "The following proposals passed consensus during this task. "
            "Use them as verified short-term shared context only for this task.\n"
            f"SHARED_TASK_MEMORY\n{json.dumps(payloads, ensure_ascii=False, default=str)}\n"
            "END_SHARED_TASK_MEMORY"
        )

    def clear(self) -> None:
        self.accepted_proposals.clear()


def _proposal_payload(proposal: Any) -> dict[str, Any]:
    if hasattr(proposal, "to_dict"):
        payload = proposal.to_dict()
    elif hasattr(proposal, "model_dump"):
        payload = proposal.model_dump()
    elif isinstance(proposal, dict):
        payload = dict(proposal)
    else:
        payload = json.loads(json.dumps(proposal, ensure_ascii=False, default=str))
    return payload if isinstance(payload, dict) else {"value": payload}


def _proposal_consensus_result(proposal: Any) -> str | None:
    verification = getattr(proposal, "verification", None)
    if verification is None and isinstance(proposal, dict):
        verification = proposal.get("verification")
    consensus_result = getattr(verification, "consensus_result", None)
    if consensus_result is None and isinstance(verification, dict):
        consensus_result = verification.get("consensus_result")
    result = getattr(consensus_result, "result", None)
    if result is None and isinstance(consensus_result, dict):
        result = consensus_result.get("result")
    return str(result) if result is not None else None


def build_task_memory_tools(task_memory: TaskMemoryBackend):
    async def search_task_memory(query: str) -> str:
        """Search accepted short-term shared memory for the current task."""
        return json.dumps(task_memory.search(query), ensure_ascii=False, default=str)

    return [search_task_memory]


__all__ = ["TaskMemoryBackend", "build_task_memory_tools"]
