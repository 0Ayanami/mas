"""Data contracts for MARBLE memory consensus infrastructure."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


VERIFICATION_DIMENSIONS = ("veracity", "rationality", "value", "security")
DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "veracity": 0.35,
    "rationality": 0.20,
    "value": 0.20,
    "security": 0.25,
}


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def to_plain_data(value: Any) -> Any:
    """Convert dataclasses and nested containers into JSON-serializable data."""
    if is_dataclass(value):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashing and signing."""
    return json.dumps(
        to_plain_data(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _validate_score(name: str, value: float) -> float:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return float(value)


def _validate_binary(name: str, value: int) -> int:
    if int(value) not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1.")
    return int(value)


@dataclass(frozen=True)
class KeyDecision:
    decision: str
    result: str


@dataclass(frozen=True)
class ProposalThoughts:
    thoughts_abstract: str = ""
    key_decisions: List[KeyDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.thoughts_abstract) > 500:
            raise ValueError("thoughts_abstract must be 500 characters or less.")


@dataclass(frozen=True)
class ProposalAction:
    type: str
    tool: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = ""
    action_id: Optional[str] = None


@dataclass(frozen=True)
class ProposalDataReference:
    source: str
    content_snippet: str
    url: str = ""
    timestamp: str = ""


@dataclass(frozen=True)
class ProposalObservation:
    type: str
    description: str
    status: str = ""


@dataclass(frozen=True)
class MemoryProposalBody:
    thoughts: Optional[ProposalThoughts] = None
    actions: List[ProposalAction] = field(default_factory=list)
    data: List[ProposalDataReference] = field(default_factory=list)
    observations: List[ProposalObservation] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            self.thoughts is None
            and not self.actions
            and not self.data
            and not self.observations
        )


@dataclass(frozen=True)
class SelfVerificationScores:
    veracity_score: float
    rationality_score: float
    value_score: float
    security_score: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "veracity_score",
            _validate_score("veracity_score", self.veracity_score),
        )
        object.__setattr__(
            self,
            "rationality_score",
            _validate_score("rationality_score", self.rationality_score),
        )
        object.__setattr__(
            self, "value_score", _validate_score("value_score", self.value_score)
        )
        object.__setattr__(
            self,
            "security_score",
            _validate_score("security_score", self.security_score),
        )

@dataclass(frozen=True)
class MultiVerificationSummary:
    weighted_scores: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.weighted_scores.items():
            if key not in VERIFICATION_DIMENSIONS:
                raise ValueError(f"Unknown verification dimension: {key}")
            _validate_score(key, value)

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_data(self)


@dataclass(frozen=True)
class ConsensusResult:
    total_weight: float
    vote_weight: float
    result: str

    def __post_init__(self) -> None:
        if self.result not in ("pass", "fail", "pending"):
            raise ValueError("result must be pass, fail, or pending.")
        if self.total_weight < 0 or self.vote_weight < 0:
            raise ValueError("Consensus weights cannot be negative.")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_data(self)


@dataclass(frozen=True)
class ProposalVerification:
    self_verification: SelfVerificationScores
    multi_verification: Optional[MultiVerificationSummary] = None
    consensus_result: Optional[ConsensusResult] = None


@dataclass(frozen=True)
class MemoryProposalHeader:
    proposal_id: str
    task_id: str
    timestamp: str
    agent_id: str
    agent_signature: str
    body_hash: str
    proposal_summary: str
    parent_proposals: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.proposal_summary) > 200:
            raise ValueError("proposal_summary must be 200 characters or less.")


@dataclass(frozen=True)
class MemoryProposal:
    header: MemoryProposalHeader
    body: MemoryProposalBody
    verification: ProposalVerification

    @property
    def proposal_id(self) -> str:
        return self.header.proposal_id

    @property
    def task_id(self) -> str:
        return self.header.task_id
    
    @property
    def timestamp(self) -> str:
        return self.header.timestamp

    @property
    def agent_id(self) -> str:
        return self.header.agent_id

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_data(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class VerificationContext:
    task_id: str
    task_description: str = ""
    related_proposals: List[MemoryProposal] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationVector:
    veracity: int
    rationality: int
    value: int
    security: int
    
    reasoning: str = ""
    verifier_agent_id: Optional[str] = None
    timestamp: str = field(default_factory=utc_now_iso)
    dimension_weights: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_DIMENSION_WEIGHTS.copy()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for dimension in VERIFICATION_DIMENSIONS:
            object.__setattr__(
                self, dimension, _validate_binary(dimension, getattr(self, dimension))
            )
        total_weight = sum(self.dimension_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("Verification dimension weights must sum to 1.")
        for dimension, weight in self.dimension_weights.items():
            if dimension not in VERIFICATION_DIMENSIONS:
                raise ValueError(f"Unknown verification dimension: {dimension}")
            if weight < 0:
                raise ValueError("Verification weights cannot be negative.")

    @property
    def confidence_score(self) -> float:
        return sum(
            getattr(self, dimension) * self.dimension_weights[dimension]
            for dimension in VERIFICATION_DIMENSIONS
        )

    def to_dict(self) -> Dict[str, Any]:
        data = to_plain_data(self)
        data["confidence_score"] = self.confidence_score
        return data


@dataclass(frozen=True)
class ConsensusVote:
    """A verifier's baseline vote derived from a verification vector."""

    proposal_id: str
    voter_agent_id: str
    accept: bool
    confidence_score: float
    verification: VerificationVector
    reason: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        _validate_score("confidence_score", self.confidence_score)
        if self.weight < 0:
            raise ValueError("vote weight cannot be negative.")

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_data(self)


@dataclass(frozen=True)
class ConsensusDecision:
    """Consensus outcome and trace for one memory proposal."""

    proposal_id: str
    result: str
    votes: List[ConsensusVote] = field(default_factory=list)
    accept_count: int = 0
    reject_count: int = 0
    acceptance_ratio: float = 0.0
    threshold: float = 0.5
    confidence_threshold: float = 0.6
    total_weight: float = 0.0
    accept_weight: float = 0.0
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.result not in ("pass", "fail", "pending"):
            raise ValueError("result must be pass, fail, or pending.")
        _validate_score("acceptance_ratio", self.acceptance_ratio)
        _validate_score("threshold", self.threshold)
        _validate_score("confidence_threshold", self.confidence_threshold)
        if self.accept_count < 0 or self.reject_count < 0:
            raise ValueError("vote counts cannot be negative.")
        if self.total_weight < 0 or self.accept_weight < 0:
            raise ValueError("vote weights cannot be negative.")

    @property
    def accepted(self) -> bool:
        return self.result == "pass"

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_data(self)
