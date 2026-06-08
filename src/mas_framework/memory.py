from __future__ import annotations

import json
import os
import sqlite3
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
        filters: dict[str, Any] = {"run_id": task_id} if task_id else {"agent_id": "*"}
        response = self.client.search(query, filters=filters, top_k=limit)
        if isinstance(response, dict):
            return list(response.get("results", []))
        return list(response or [])


class SQLiteMemoryStore:
    def __init__(
        self,
        db_path: str | Path = "data/memory.sqlite",
        log_path: str | Path | None = None,
        *,
        mem0_backend: Mem0MemoryBackend | None = None,
        enable_mem0: bool | None = None,
    ):
        self.db_path = Path(db_path)
        self.log_path = Path(log_path) if log_path is not None else self.db_path.with_name("audit.jsonl")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.mem0_backend = mem0_backend
        if self.mem0_backend is None and (enable_mem0 or os.getenv("MAS_ENABLE_MEM0") == "1"):
            self.mem0_backend = Mem0MemoryBackend(api_key=os.getenv("MEM0_API_KEY"))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proposals (
                    proposal_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_proposals_task_status
                ON proposals(task_id, status)
                """
            )

    def save_proposal(self, proposal: MemoryProposal) -> None:
        payload = proposal.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO proposals (
                    proposal_id, content_hash, agent_id, task_id, memory_type,
                    title, status, payload, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.proposal_id,
                    proposal.content_hash,
                    proposal.agent_id,
                    proposal.task_id,
                    proposal.memory_type,
                    proposal.title,
                    proposal.status.value,
                    payload,
                    proposal.timestamp.isoformat(),
                ),
            )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        if self.mem0_backend is not None:
            self.mem0_backend.add_proposal(proposal)

    def get_proposal(self, proposal_id: str) -> MemoryProposal:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            raise KeyError(proposal_id)
        return MemoryProposal.model_validate_json(row["payload"])

    def list_proposals(
        self,
        *,
        task_id: str | None = None,
        status: ProposalStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryProposal]:
        clauses = []
        params: list[str | int] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT payload FROM proposals {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [MemoryProposal.model_validate_json(row["payload"]) for row in rows]

    def search(self, query: str, limit: int = 10) -> list[MemoryProposal]:
        like = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM proposals
                WHERE title LIKE ? OR payload LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (like, like, limit),
            ).fetchall()
        return [MemoryProposal.model_validate_json(row["payload"]) for row in rows]

    def export_json(self, path: str | Path) -> None:
        proposals = [proposal.model_dump(mode="json") for proposal in self.list_proposals(limit=10000)]
        Path(path).write_text(json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8")
