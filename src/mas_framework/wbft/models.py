"""Data contracts for WBFT agent-output consensus."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


WBFTMode = Literal["round_robin", "magentic_one"]


def to_plain_data(value: Any) -> Any:
    """Convert dataclasses and nested containers into JSON-serializable data."""
    if is_dataclass(value):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


@dataclass(frozen=True)
class WBFTAgentSpec:
    agent_id: str
    display_name: str
    model: str | None
    is_byzantine: bool = False

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self)


@dataclass(frozen=True)
class WBFTAgentResponse:
    agent_id: str
    answer: str
    confidence: float
    reasoning: str = ""
    raw_content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        confidence = max(0.0, min(1.0, float(self.confidence)))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "answer", str(self.answer))

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self)


@dataclass(frozen=True)
class WBFTConsensusResult:
    consensus_answer: str
    consensus_confidence: float
    participant_count: int
    convergence_achieved: bool
    responses: list[WBFTAgentResponse] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        confidence = max(0.0, min(1.0, float(self.consensus_confidence)))
        object.__setattr__(self, "consensus_confidence", confidence)

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self)


@dataclass
class WBFTRunConfig:
    mode: WBFTMode = "magentic_one"
    max_messages: int = 50
    model: str | None = None
    honest_model: str | None = None
    byzantine_model: str | None = None

    confidence_extraction_method: Literal["json", "regex"] = "regex"
    confidence_instruction: str = (
        "Confidence Assessment Guidelines:\n"
        "0.0-0.2: Very uncertain - multiple possibilities, unclear reasoning, or major doubts.\n"
        "0.2-0.4: Low confidence - leaning toward an answer but with significant uncertainty.\n"
        "0.4-0.6: Moderate confidence - reasonably sure but still possible alternatives.\n"
        "0.6-0.8: High confidence - clear reasoning with minor uncertainty.\n"
        "0.8-1.0: Very high confidence - strongly supported and carefully checked.\n"
        "Please honestly assess your confidence based on task evidence and reasoning clarity."
    )
    confidence_threshold: float = 0.0
    convergence_threshold: float = 0.0
    fault_tolerance_threshold: float = 0.33
    minimum_participants: int = 1
    normalization: Literal["auto", "number", "none"] = "auto"
    include_unstructured_outputs: bool = False
    fallback_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.confidence_extraction_method not in {"json", "regex"}:
            raise ValueError("confidence_extraction_method must be json or regex.")

    @classmethod
    def from_config(
        cls,
        path: str,
        *,
        mode: WBFTMode | None = None,
    ) -> "WBFTRunConfig":
        import yaml
        from pathlib import Path

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        agents = payload.get("agents", {})
        collaboration = payload.get("collaboration", {})
        wbft = payload.get("wbft", {})
        models = agents.get("models", {})
        default_model = agents.get("model") or models.get("honest")

        return cls(
            mode=mode or collaboration.get("mode", "magentic_one"),
            max_messages=int(collaboration.get("max_messages", collaboration.get("rounds", 50))),
            model=default_model,
            honest_model=models.get("honest", default_model),
            byzantine_model=models.get("byzantine", default_model),
            confidence_extraction_method=wbft.get("confidence_extraction_method", "regex"),
            confidence_instruction=wbft.get(
                "confidence_instruction",
                cls.confidence_instruction,
            ),
            confidence_threshold=float(wbft.get("confidence_threshold", 0.0)),
            convergence_threshold=float(wbft.get("convergence_threshold", 0.0)),
            fault_tolerance_threshold=float(wbft.get("fault_tolerance_threshold", 0.33)),
            minimum_participants=int(wbft.get("minimum_participants", 1)),
            normalization=wbft.get("normalization", "auto"),
            include_unstructured_outputs=bool(wbft.get("include_unstructured_outputs", False)),
            fallback_confidence=float(wbft.get("fallback_confidence", 0.0)),
        )

    @classmethod
    def from_tamas_unified_config(
        cls,
        path: str,
        *,
        mode: WBFTMode | None = None,
    ) -> "WBFTRunConfig":
        """Backward-compatible alias; prefer from_config with wbft_config.yaml."""
        return cls.from_config(path, mode=mode)
