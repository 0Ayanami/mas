from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from camel.types import ModelPlatformType, ModelType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mas_framework.memory import Mem0MemoryBackend
    from mas_framework.tools import ToolRegistry

class ProposalStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"

@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return asdict(self)

@dataclass
class AgentConfig:
    agent_id: str
    model_platform: ModelPlatformType = ModelPlatformType.DEFAULT
    model_type: ModelType = ModelType.DEFAULT
    model_config_dict: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    role: str = ""
    system_prompt: str = ""
    memory: "Mem0MemoryBackend | None" = None
    tools: "ToolRegistry | None" = None

@dataclass
class VerificationVector:
    veracity: int
    rationality: int
    value: int
    security: int
    confidence: float
    rationale: str
    verifier_id: str
    weight: float = 1.0

    """
    Veracity：主要面向Data和Observation字段，针对给出的事实性信息进行真实性、准确性判断。全部条件通过记为1，否则0。
    Rationality：主要面向Thoughts和Action字段，针对思维链决策链判断合理性，动作执行（工具选择、执行逻辑）合理性等。全部条件通过记为1，否则0。
    Value ：主要面向Data和Observation字段，判断与主线任务是否相关，对其他agent工作是否有支撑作用等。全部条件通过记为1，否则0。
    Security：对几个字段进行综合判定，判断是否出现了常见的拜占庭模式（poison、injection、hallucination等）。无显著安全风险记为1，否则0。
    """
    def __post_init__(self) -> None:
        for name in ["veracity", "rationality", "value", "security"]:
            value = getattr(self, name)
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @classmethod
    def from_binary_votes(
        cls,
        *,
        veracity: bool | None = None,
        rationality: bool | None = None,
        value: bool | None = None,
        security: bool | None = None,
        rationale: str,
        verifier_id: str,
        weights: dict[str, float] = {
            "veracity": 0.30,
            "rationality": 0.25,
            "value": 0.25,
            "security": 0.20,
        },
    ) -> VerificationVector:
        votes = {
            "veracity": int(veracity),
            "rationality": int(rationality),
            "value": int(value),
            "security": int(security),
        }
        confidence = sum(votes[key] * weights[key] for key in votes)
        return cls(
            **votes,
            confidence=round(confidence, 4),
            rationale=rationale,
            verifier_id=verifier_id,
        )

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return asdict(self)

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass
class ProposalHeader:
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    timestamp: datetime = field(default_factory=_utc_now)
    proposing_agent_id: str = ""
    proposing_agent_signature: str = ""
    parent_proposal_ids: list[str] = field(default_factory=list)
    body_hash: str = ""
    proposal_summary: str = ""
    memory_type: Literal["research_note", "evidence", "milestone", "tool_observation"] = "research_note"

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        self.parent_proposal_ids = list(self.parent_proposal_ids or [])
        if not self.proposing_agent_signature and self.proposing_agent_id:
            self.proposing_agent_signature = self.proposing_agent_id

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass
class ProposalBody:
    thoughts: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    data: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.thoughts = dict(self.thoughts or {})
        self.actions = [
            item if isinstance(item, dict) else {"description": str(item)}
            for item in _as_list(self.actions)
        ]
        self.data = [
            item if isinstance(item, dict) else {"content": item}
            for item in _as_list(self.data)
        ]
        self.observations = [
            item if isinstance(item, dict) else {"description": str(item)}
            for item in _as_list(self.observations)
        ]

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelfVerification:
    """
    对准备propose的memory进行自我验证
    """
    veracity: int = 1
    rationality: int = 1
    value: int = 1
    security: int = 1
    confidence: float = 0.0
    rationale: str = ""

    def __post_init__(self) -> None:
        for name in ["veracity", "rationality", "value", "security"]:
            value = getattr(self, name)
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiAgentVerificationSummary:
    """
    对agent的验证结果进行汇总，将各个agent的维度打分进行加权平均，得到一个整体的验证结果
    """
    veracity: float | None = None
    rationality: float | None = None
    value: float | None = None
    security: float | None = None
    confidence: float | None = None
    verifier_count: int = 0

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConsensusResult:
    voting_agents: int = 0
    total_agents: int = 0
    vote_weight: float = 0.0
    total_weight: float = 0.0
    result: ProposalStatus = ProposalStatus.PENDING

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProposalVerification:
    self_verification: SelfVerification = field(default_factory=SelfVerification)
    multi_agent_verification: MultiAgentVerificationSummary = field(
        default_factory=MultiAgentVerificationSummary
    )
    consensus_result: ConsensusResult = field(default_factory=ConsensusResult)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {
            "self_verification": self.self_verification.model_dump(mode=mode),
            "multi_agent_verification": self.multi_agent_verification.model_dump(mode=mode),
            "consensus_result": self.consensus_result.model_dump(mode=mode),
        }


@dataclass(init=False)
class MemoryProposal:
    header: ProposalHeader
    body: ProposalBody
    verification: ProposalVerification
    status: ProposalStatus = ProposalStatus.PENDING
    consensus_round: int = 0
    verifications: list[VerificationVector]

    def __init__(
        self,
        *,
        header: ProposalHeader | dict[str, Any] | None = None,
        body: ProposalBody | dict[str, Any] | None = None,
        verification: ProposalVerification | dict[str, Any] | None = None,
        status: ProposalStatus | str = ProposalStatus.PENDING,
        consensus_round: int = 0,
        verifications: list[VerificationVector | dict[str, Any]] | None = None,

        agent_id: str | None = None,
        task_id: str | None = None,
        memory_type: Literal["research_note", "evidence", "milestone", "tool_observation"] = "research_note",
        title: str | None = None,
        thoughts_decision: str | None = None,
        action: str | list[Any] | None = None,
        results_observation: str | list[Any] | None = None,
        self_confidence: float | None = None,
        data: dict[str, Any] | list[Any] | None = None,
        proposal_id: str | None = None,
        timestamp: datetime | str | None = None,
        parent_proposal_ids: list[str] | None = None,
        proposing_agent_signature: str | None = None,
    ) -> None:
        self.status = ProposalStatus(status)
        self.consensus_round = consensus_round
        self.verifications = [
            item if isinstance(item, VerificationVector) else VerificationVector(**item)
            for item in (verifications or [])
        ]

        if header is None:
            header = ProposalHeader(
                proposal_id=proposal_id or str(uuid.uuid4()),
                task_id=task_id or "",
                timestamp=timestamp or _utc_now(),
                proposing_agent_id=agent_id or "",
                proposing_agent_signature=proposing_agent_signature or agent_id or "",
                parent_proposal_ids=parent_proposal_ids or [],
                proposal_summary=title or "",
                memory_type=memory_type,
            )
        elif isinstance(header, dict):
            header = ProposalHeader(**header)

        if body is None:
            action_items = action if isinstance(action, list) else _as_list(action)
            observation_items = (
                results_observation
                if isinstance(results_observation, list)
                else _as_list(results_observation)
            )
            body = ProposalBody(
                thoughts={
                    "thoughts_abstract": thoughts_decision or "",
                    "key_decision_points": [],
                },
                actions=[
                    item if isinstance(item, dict) else {"action_id": f"action_{idx}", "description": str(item)}
                    for idx, item in enumerate(action_items, start=1)
                    if item is not None
                ],
                data=self._normalize_data(data),
                observations=[
                    item
                    if isinstance(item, dict)
                    else {"observation_id": f"result_{idx}", "description": str(item)}
                    for idx, item in enumerate(observation_items, start=1)
                    if item is not None
                ],
            )
        elif isinstance(body, dict):
            body = ProposalBody(**body)

        if verification is None:
            verification = ProposalVerification(
                self_verification=SelfVerification(confidence=self_confidence or 0.0)
            )
        elif isinstance(verification, dict):
            verification = ProposalVerification(
                self_verification=(
                    verification.get("self_verification")
                    if isinstance(verification.get("self_verification"), SelfVerification)
                    else SelfVerification(**verification.get("self_verification", {}))
                ),
                multi_agent_verification=(
                    verification.get("multi_agent_verification")
                    if isinstance(verification.get("multi_agent_verification"), MultiAgentVerificationSummary)
                    else MultiAgentVerificationSummary(**verification.get("multi_agent_verification", {}))
                ),
                consensus_result=(
                    verification.get("consensus_result")
                    if isinstance(verification.get("consensus_result"), ConsensusResult)
                    else ConsensusResult(**verification.get("consensus_result", {}))
                ),
            )

        self.header = header
        self.body = body
        self.verification = verification
        self.header.body_hash = self.body_hash

    @staticmethod
    def _normalize_data(data: dict[str, Any] | list[Any] | None) -> list[dict[str, Any]]:
        if data is None:
            return []
        if isinstance(data, list):
            return [item if isinstance(item, dict) else {"content": item} for item in data]
        return [{"data_id": key, "content": value} for key, value in data.items()]

    @property
    def agent_id(self) -> str:
        return self.header.proposing_agent_id

    @property
    def task_id(self) -> str:
        return self.header.task_id

    @property
    def memory_type(self) -> str:
        return self.header.memory_type

    @property
    def title(self) -> str:
        return self.header.proposal_summary

    @property
    def proposal_id(self) -> str:
        return self.header.proposal_id

    @property
    def timestamp(self) -> datetime:
        return self.header.timestamp

    @property
    def self_confidence(self) -> float:
        return self.verification.self_verification.confidence

    @property
    def thoughts_decision(self) -> str:
        return str(self.body.thoughts.get("thoughts_abstract", ""))

    @property
    def action(self) -> str:
        return "; ".join(str(item.get("description", item)) for item in self.body.actions)

    @property
    def data(self) -> dict[str, Any]:
        return {str(item.get("data_id", idx)): item.get("content", item) for idx, item in enumerate(self.body.data)}

    @property
    def results_observation(self) -> str:
        return "; ".join(str(item.get("description", item)) for item in self.body.observations)

    @property
    def body_hash(self) -> str:
        payload = self.body.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def content_hash(self) -> str:
        return self.body_hash

    def short_label(self) -> str:
        return f"{self.title} ({self.proposal_id[:8]})"

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        self.header.body_hash = self.body_hash
        return {
            "header": self.header.model_dump(mode=mode),
            "body": self.body.model_dump(mode=mode),
            "verification": self.verification.model_dump(mode=mode),
            "status": self.status.value,
            "consensus_round": self.consensus_round,
            "verifications": [item.model_dump(mode=mode) for item in self.verifications],
        }

    def model_dump_json(self, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=indent)

    @classmethod
    def model_validate_json(cls, payload: str) -> "MemoryProposal":
        return cls(**json.loads(payload))


@dataclass
class ConsensusDecision:
    proposal_id: str
    status: ProposalStatus
    quorum_size: int
    positive_votes: float
    average_confidence: float
    threshold: float
    rationale: str

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ProposalStatus(self.status)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
