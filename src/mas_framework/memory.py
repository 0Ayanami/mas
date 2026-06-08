from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mas_framework.models import MemoryProposal, ProposalStatus


class Mem0MemoryBackend:
    def __init__(self, *, api_key: str | None = None, client: Any | None = None):
        self.client = client
        if self.client is None:
            os.environ.setdefault("MEM0_DIR", str(Path("data/mem0").resolve()))
            from mem0 import MemoryClient

            self.client = MemoryClient(api_key=api_key)

    def add_proposal(self, proposal: MemoryProposal) -> Any:
        content = proposal.model_dump_json(indent=2)
        metadata = {
            "proposal_id": proposal.proposal_id,
            "task_id": proposal.task_id,
            "agent_id": proposal.agent_id,
            "status": proposal.status.value,
            "content_hash": proposal.content_hash,
            "memory_type": proposal.memory_type,
        }
        return self.client.add(
            [{"role": "assistant", "content": content}],
            agent_id=proposal.agent_id,
            run_id=proposal.task_id,
            metadata=metadata,
            infer=False,
        )

    def search(self, query: str, *, task_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {"task_id": task_id} if task_id else {"agent_id": "*"}
        response = self.client.search(query, filters=filters, top_k=limit)
        if isinstance(response, dict):
            return list(response.get("results", []))
        return list(response or [])
