from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mas_framework.models import MemoryProposal
    
class Mem0MemoryBackend:
    def __init__(self, *, api_key: str | None = None, client: Any | None = None, topk:int = 5):
        self.client = client
        self.topk = topk
        if self.client is None:
            os.environ.setdefault("MEM0_DIR", str(Path("data/mem0").resolve()))
            from mem0 import MemoryClient

            self.client = MemoryClient(api_key=api_key)

    def add_proposal(self, proposal: MemoryProposal, user_id: str) -> Any:
        content = proposal.model_dump_json(indent=2)
        metadata = {
            "proposal_id": proposal.header.proposal_id,
            "task_id": proposal.header.task_id,
            "agent_id": proposal.header.proposing_agent_id,
            "agent_signature": proposal.header.proposing_agent_signature,
        }
        return self.client.add(
            [{"role": "assistant", "content": content}],
            user_id=user_id,
            metadata=metadata,
        )

    def search(self, query: str, user_id: str | None = None) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {"user_id": user_id} if user_id else {}
        response = self.client.search(query, filters=filters, top_k=self.topk)
        if not response:
            return "No memory records matched the query."
        if isinstance(response, dict):
            return list(response.get("results", []))
        return list(response)

    def update_proposal(self, memory_id: str, proposal: MemoryProposal, user_id: str) -> Any:
        text = proposal.model_dump_json(indent=2)
        metadata = {
            "proposal_id": proposal.header.proposal_id,
            "task_id": proposal.header.task_id,
            "agent_id": proposal.header.proposing_agent_id,
            "agent_signature": proposal.header.proposing_agent_signature,
        }
        try:
            return self.client.update(memory_id, text=text, metadata=metadata)
        except Exception:
            return self.add_proposal(proposal, user_id=user_id)

