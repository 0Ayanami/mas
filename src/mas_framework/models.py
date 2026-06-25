from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
class ProposalStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"

@dataclass
class AgentConfig:
    agent_id: str
    role: str = ""
    system_prompt: str = ""
    conf_threshold: float = 0.7
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    model_config_dict: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    model_info: dict[str, Any] | None = None
    max_tool_iterations: int = 10
    reflect_on_tool_use: bool = True
    model_client: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = (
                os.getenv("AUTOGEN_MODEL")
                or os.getenv("DEFAULT_MODEL_TYPE")
                or "gpt-4o-mini"
            )

@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return asdict(self)

@dataclass
class AgentState:
    proposal_sum:int = 0
    proposal_submitted:int = 0
    base: float = 1.0
    verified_conf: float = 0.0
    verified_conf_sum: float = 0.0
    historical_conf: float = 0.0
    weight: float = 1.0
    
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return asdict(self)

@dataclass
class VerificationVector:
    # 四维分数
    veracity: int
    rationality: int
    value: int
    security: int
    # 加权计算的置信度
    confidence: float
    # 理由
    rationale: str
    # 验证agent id
    verifier_id: str
    # 置信度阈值
    conf_threshold: float
    # 投票结果
    vote_result: bool = False
    # agent权重
    weight: float = 1.0

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
        rationale: str = "",
        verifier_id: str,
        conf_threshold: float = 0.7,
        weight: float = 1.0,
        weights: dict[str, float] = {
            "veracity": 0.30,
            "rationality": 0.25,
            "value": 0.25,
            "security": 0.20,
        },
    ) -> VerificationVector:
        votes = {
            "veracity": int(veracity or 0),
            "rationality": int(rationality or 0),
            "value": int(value or 0),
            "security": int(security or 0),
        }
        confidence = sum(votes[key] * weights[key] for key in votes)
        vote_result = confidence >= conf_threshold
        return cls(
            **votes,
            confidence=round(confidence, 4),
            rationale=rationale,
            verifier_id=verifier_id,
            vote_result=vote_result,
            conf_threshold=conf_threshold,
            weight=weight,
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


@dataclass(init=False)
class ProposalHeader:
    proposal_id: str
    # 全局唯一标识，用于唯一标识一个proposal
    task_id: str
    # 所属任务ID，关联上级任务
    
    timestamp: datetime
    # ISO8601格式的时间戳
    agent_id: str
    agent_signature: str
    # 对Body内容的数字签名

    parent_proposals: list[str]
    # 父提案ID列表（引用关系）
    body_hash: str
    # Body内容的SHA-256哈希
    
    proposal_summary: str
    # 一句话摘要，用于快速检索
    memory_type: Literal["research_note", "evidence", "milestone", "tool_observation"]
    # 内存类型，用于分类

    def __init__(
        self,
        *,
        proposal_id: str | None = None,
        task_id: str = "",
        timestamp: datetime | str | None = None,
        agent_id: str | None = None,
        agent_signature: str | None = None,
        parent_proposals: list[str] | None = None,
        body_hash: str = "",
        proposal_summary: str = "",
        memory_type: Literal["research_note", "evidence", "milestone", "tool_observation"] = "research_note",
        proposing_agent_id: str | None = None,
        proposing_agent_signature: str | None = None,
        parent_proposal_ids: list[str] | None = None,
    ) -> None:
        self.proposal_id = proposal_id or str(uuid.uuid4())
        self.task_id = task_id
        self.timestamp = timestamp or _utc_now()
        self.agent_id = agent_id or proposing_agent_id or ""
        self.agent_signature = agent_signature or proposing_agent_signature or self.agent_id
        self.parent_proposals = list(parent_proposals or parent_proposal_ids or [])
        self.body_hash = body_hash
        self.proposal_summary = proposal_summary
        self.memory_type = memory_type
        self.__post_init__()
    
    def __post_init__(self) -> None:
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        self.parent_proposals = list(self.parent_proposals or [])
        if not self.agent_signature:
            self.agent_signature = self.agent_id

    @property
    def proposing_agent_id(self) -> str:
        return self.agent_id

    @proposing_agent_id.setter
    def proposing_agent_id(self, value: str) -> None:
        self.agent_id = value

    @property
    def proposing_agent_signature(self) -> str:
        return self.agent_signature

    @proposing_agent_signature.setter
    def proposing_agent_signature(self, value: str) -> None:
        self.agent_signature = value

    @property
    def parent_proposal_ids(self) -> list[str]:
        return self.parent_proposals

    @parent_proposal_ids.setter
    def parent_proposal_ids(self, value: list[str]) -> None:
        self.parent_proposals = list(value or [])

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        timestamp = self.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return {
            "proposal_id": self.proposal_id,
            "task_id": self.task_id,
            "timestamp": timestamp.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "agent_id": self.agent_id,
            "agent_signature": self.agent_signature,
            "parent_proposals": list(self.parent_proposals),
            "body_hash": self.body_hash,
            "proposal_summary": self.proposal_summary,
        }


@dataclass
class ProposalBody:
    thoughts: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    data: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.thoughts = dict(self.thoughts or {})
        if "key_decision_points" in self.thoughts and "key_decisions" not in self.thoughts:
            self.thoughts["key_decisions"] = self.thoughts.pop("key_decision_points")

        self.actions = [
            item if isinstance(item, dict) else {"description": str(item)}
            for item in _as_list(self.actions)
        ]

        normalized_data = []
        for item in _as_list(self.data):
            if not isinstance(item, dict):
                normalized_data.append({"content_snippet": str(item)})
                continue
            item = dict(item)
            if "content" in item and "content_snippet" not in item:
                item["content_snippet"] = item.pop("content")
            normalized_data.append(item)
        self.data = normalized_data

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
    veracity: int | None = None
    rationality: int | None = None
    value: int | None = None
    security: int | None = None
    confidence: float | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        for name in ["veracity", "rationality", "value", "security"]:
            value = getattr(self, name)
            if value is not None and value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
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
    voting_agents: int | None = None
    total_agents: int | None = None
    vote_weight: float | None = None
    total_weight: float | None = None
    result: ProposalStatus = ProposalStatus.PENDING

    def __post_init__(self) -> None:
        if isinstance(self.result, str):
            self.result = ProposalStatus(self.result)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        payload = asdict(self)
        payload["result"] = self.result.value
        return payload


@dataclass
class ProposalVerification:
    self_verification: SelfVerification | None = None
    multi_agent_verification: MultiAgentVerificationSummary | None = None

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {
            "self_verification": (
                self.self_verification.model_dump(mode=mode)
                if self.self_verification is not None
                else None
            ),
            "multi_agent_verification": (
                self.multi_agent_verification.model_dump(mode=mode)
                if self.multi_agent_verification is not None
                else None
            ),
        }


@dataclass(init=False)
class MemoryProposal:
    header: ProposalHeader
    body: ProposalBody
    verification: ProposalVerification
    consensus_result: ConsensusResult
    status: ProposalStatus = ProposalStatus.PENDING
    consensus_round: int = 0
    verifications: list[VerificationVector]

    def __init__(
        self,
        *,
        header: ProposalHeader | dict[str, Any] | None = None,
        body: ProposalBody | dict[str, Any] | None = None,
        verification: ProposalVerification | dict[str, Any] | None = None,
        consensus_result: ConsensusResult | dict[str, Any] | None = None,

        status: ProposalStatus | str = ProposalStatus.PENDING,
        consensus_round: int = 0,
        verifications: list[VerificationVector | dict[str, Any]] | None = None,
        
        # Header
        agent_id: str | None = None,
        task_id: str | None = None,
        memory_type: Literal["research_note", "evidence", "milestone", "tool_observation"] = "research_note",
        summary: str | None = None,
        title: str | None = None,
        proposal_id: str | None = None,
        timestamp: datetime | str | None = None,
        parent_proposal_ids: list[str] | None = None,
        proposing_agent_signature: str | None = None,

        # Body
        thoughts_decision: str | None = None,
        action: str | list[Any] | None = None,
        results_observation: str | list[Any] | None = None,
        data: dict[str, Any] | list[Any] | None = None,
        self_confidence: float | None = None,
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
                proposal_summary=summary or title or "",
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
                    "key_decisions": [],
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

        if isinstance(verification, dict):
            self_verification = verification.get("self_verification")
            multi_agent_verification = verification.get("multi_agent_verification")
            verification = ProposalVerification(
                self_verification=(
                    self_verification
                    if isinstance(self_verification, SelfVerification)
                    else SelfVerification(**self_verification)
                    if isinstance(self_verification, dict)
                    else None
                ),
                multi_agent_verification=(
                    multi_agent_verification
                    if isinstance(multi_agent_verification, MultiAgentVerificationSummary)
                    else MultiAgentVerificationSummary(**multi_agent_verification)
                    if isinstance(multi_agent_verification, dict)
                    else None
                ),
            )
        elif verification is None:
            verification = ProposalVerification(
                self_verification=(
                    SelfVerification(confidence=self_confidence)
                    if self_confidence is not None
                    else None
                )
            )

        if isinstance(consensus_result, dict):
            consensus_result = ConsensusResult(**consensus_result)
        elif consensus_result is None:
            consensus_result = ConsensusResult()

        self.header = header
        self.body = body
        self.verification = verification
        self.header.body_hash = self.body_hash
        self.consensus_result = consensus_result

    @staticmethod
    def _normalize_data(data: dict[str, Any] | list[Any] | None) -> list[dict[str, Any]]:
        if data is None:
            return []
        if isinstance(data, list):
            return [item if isinstance(item, dict) else {"content": item} for item in data]
        return [{"source": key, "content_snippet": value} for key, value in data.items()]

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
    def summary(self) -> str:
        return self.header.proposal_summary

    @property
    def title(self) -> str:
        return self.summary

    @property
    def proposal_id(self) -> str:
        return self.header.proposal_id

    @property
    def timestamp(self) -> datetime:
        return self.header.timestamp

    @property
    def self_confidence(self) -> float | None:
        if self.verification.self_verification is None:
            return None
        return self.verification.self_verification.confidence
    
    @property
    def multi_confidence(self) -> float | None:
        if self.verification.multi_agent_verification is None:
            return None
        return self.verification.multi_agent_verification.confidence

    @property
    def thoughts_decision(self) -> str:
        return str(self.body.thoughts.get("thoughts_abstract", ""))

    @property
    def action(self) -> str:
        return "; ".join(str(item.get("description", item)) for item in self.body.actions)

    @property
    def data(self) -> dict[str, Any]:
        return {
            str(item.get("data_id", item.get("source", idx))): item.get(
                "content_snippet", item.get("content", item)
            )
            for idx, item in enumerate(self.body.data)
        }

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
        return f"{self.summary} ({self.proposal_id[:8]})"

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        self.header.body_hash = self.body_hash
        return {
            "header": self.header.model_dump(mode=mode),
            "body": self.body.model_dump(mode=mode),
            "verification": self.verification.model_dump(mode=mode),
            "consensus_result": self.consensus_result.model_dump(mode=mode),
            "status": self.status.value,
            "consensus_round": self.consensus_round,
            "verifications": [item.model_dump(mode=mode) for item in self.verifications],
        }

    def model_dump_json(self, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=indent)

    @classmethod
    def model_validate_json(cls, payload: str) -> "MemoryProposal":
        return cls(**json.loads(payload))
