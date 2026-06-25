from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mas_framework.models import MemoryProposal
    
class Mem0MemoryBackend:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        topk: int = 5,
        default_user_id: str = "shared_mas",
    ):
        self.client = client
        self.topk = topk
        self.default_user_id = default_user_id
        if self.client is None:
            os.environ.setdefault("MEM0_DIR", str(Path("data/mem0").resolve()))
            from mem0 import MemoryClient

            self.client = MemoryClient(api_key=api_key)

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

    def add_proposal(self, proposal: MemoryProposal, user_id: str) -> Any:
        """Adds a negotiation proposal to the memory.

        Args:
            proposal: The MemoryProposal object to be stored.
            user_id: The ID of the user associated with this memory entry.

        Returns:
            The result from the underlying memory client's add operation.
        """
        content = proposal.model_dump_json(indent=2)
        metadata = {
            "proposal_id": proposal.header.proposal_id,
            "task_id": proposal.header.task_id,
            "agent_id": proposal.header.agent_id,
            "agent_signature": proposal.header.agent_signature,
            "parent_proposals": proposal.header.parent_proposals,
            "body_hash": proposal.header.body_hash,
        }
        return self.add(content, user_id=user_id, metadata=metadata)

    def search(self, query: str, user_id: str | None = None) -> list[dict[str, Any]]:
        """Searches the memory for a given query.

        Args:
            query: The search query string.
            user_id: Optional ID of the user to scope the search. If not provided,
                the default user ID for the session is used.

        Returns:
            A list of search results, where each result is a dictionary.
        """
        filters: dict[str, Any] = {"user_id": user_id or self.default_user_id}
        response = self.client.search(query, filters=filters, top_k=self.topk)
        if not response:
            return []
        if isinstance(response, dict):
            return list(response.get("results", []))
        return list(response)

    def update(
        self,
        memory_id: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return self.client.update(memory_id, text=content, metadata=metadata or {})

    def update_proposal(self, memory_id: str, proposal: MemoryProposal, user_id: str) -> Any:
        """Updates a proposal in memory, or adds it if the update fails.

        This method first attempts to update an existing memory entry identified by
        `memory_id`. If the update operation fails for any reason (e.g., the
        ID is not found), it falls back to adding the proposal as a new entry.

        Args:
            memory_id: The unique identifier of the memory entry to update.
            proposal: The `MemoryProposal` object with the updated content.
            user_id: The ID of the user. Used for the fallback `add_proposal` call.

        Returns:
            The result from the underlying memory client's update or add operation.
        """
        text = proposal.model_dump_json(indent=2)
        metadata = {
            "proposal_id": proposal.header.proposal_id,
            "task_id": proposal.header.task_id,
            "agent_id": proposal.header.agent_id,
            "agent_signature": proposal.header.agent_signature,
            "parent_proposals": proposal.header.parent_proposals,
            "body_hash": proposal.header.body_hash,
        }
        try:
            return self.update(memory_id, text, metadata=metadata)
        except Exception:
            return self.add_proposal(proposal, user_id=user_id)
