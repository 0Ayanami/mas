from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class Mem0MemoryBackend:
    """Mem0-backed memory store used by AutoGen agents in TAMAS experiments."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        topk: int = 5,
        default_user_id: str = "shared_mas",
    ) -> None:
        self.client = client
        self.topk = topk
        self.default_user_id = default_user_id
        if self.client is None:
            load_dotenv()
            os.environ.setdefault("MEM0_DIR", str(Path("data/mem0").resolve()))
            from mem0 import MemoryClient

            self.client = MemoryClient(api_key=api_key or os.getenv("MEM0_API_KEY"))

    def add(
        self,
        content: str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return self.client.add(
            [{"role": "assistant", "content": content}],
            user_id=user_id or self.default_user_id,
            metadata=metadata or {},
        )

    def search(self, query: str, user_id: str | None = None) -> list[dict[str, Any]]:
        response = self.client.search(
            query,
            filters={"user_id": user_id or self.default_user_id},
            top_k=self.topk,
        )
        if not response:
            return []
        if isinstance(response, dict):
            return list(response.get("results", []))
        return list(response)

    def add_proposal(self, proposal: Any, user_id: str | None = None) -> Any:
        consensus_result = _proposal_consensus_result(proposal)
        if consensus_result != "pass":
            raise ValueError(
                "Only memory proposals with consensus_result='pass' can be uploaded to Mem0."
            )
        return self.add(
            _proposal_json(proposal),
            user_id=user_id,
            metadata=_proposal_metadata(proposal),
        )


def _proposal_json(proposal: Any) -> str:
    if hasattr(proposal, "to_json"):
        return proposal.to_json()
    if hasattr(proposal, "model_dump_json"):
        return proposal.model_dump_json(indent=2)
    return json.dumps(proposal, ensure_ascii=False, default=str)


def _proposal_metadata(proposal: Any) -> dict[str, Any]:
    header = getattr(proposal, "header", None)
    consensus_result = _proposal_consensus_result(proposal)
    if header is None:
        return {"consensus_result": consensus_result or ""}
    return {
        "proposal_id": getattr(header, "proposal_id", ""),
        "task_id": getattr(header, "task_id", ""),
        "agent_id": getattr(header, "agent_id", ""),
        "body_hash": getattr(header, "body_hash", ""),
        "timestamp": getattr(header, "timestamp", ""),
        "consensus_result": consensus_result or "",
    }


def _proposal_consensus_result(proposal: Any) -> str | None:
    verification = getattr(proposal, "verification", None)
    consensus_result = getattr(verification, "consensus_result", None)
    return getattr(consensus_result, "result", None)


def build_memory_tools(memory_backend: Mem0MemoryBackend, *, user_id: str | None = None):
    async def search_memory(query: str) -> str:
        """Search the shared Mem0 memory for relevant prior information."""
        return json.dumps(memory_backend.search(query, user_id=user_id), ensure_ascii=False, default=str)

    return [search_memory]


__all__ = ["Mem0MemoryBackend", "build_memory_tools"]
