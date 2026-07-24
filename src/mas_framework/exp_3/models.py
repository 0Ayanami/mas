import yaml
from typing import Any, Literal
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


Exp3Method = Literal[
    "swarm",
    "swarm_cp_wbft",
    "swarm_consensus",
]

ModelRegime = Literal[
    "same",
    "weak_byzantine_strong_honest",
    "strong_byzantine_weak_honest",
]

@dataclass(frozen=True)
class AgentRuntimeSpec:
    agent_id: str
    display_name: str
    model: str | None
    capability_coefficient: float
    is_byzantine: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class TamasGroupSpec:
    method: Exp3Method
    n: int
    f: int
    model_regime: ModelRegime

    @property
    def mode(self) -> Literal["discussion_based"]:
        return "discussion_based"

    @property
    def uses_wbft(self) -> bool:
        return self.method.endswith("_cp_wbft")

    @property
    def uses_consensus(self) -> bool:
        return self.method.endswith("_consensus")

    @property
    def group_id(self) -> str:
        return f"{self.method}__n{self.n}__f{self.f}__{self.model_regime}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
@dataclass(frozen=True)
class TAMASRunConfig:
    config_path: str
    group_spec: TamasGroupSpec | None = None
    output_root: str = "experiments/exp3_tamas"
    run_id: str = "exp3_tamas"
    dataset_path: str | None = None
    methods: list[str] = field(default_factory=list)
    default_model: str = "qwen3.6-35b-a3b"
    honest_model: str = "qwen3.6-35b-a3b"
    byzantine_model: str = "qwen3.6-35b-a3b"
    temperature: float = 0.0
    capability_coefficient: float = 1.0
    capability_coefficients: dict[str, float] = field(default_factory=dict)
    max_messages: int = 6
    timeout_seconds: float = 300.0
    request_timeout_seconds: float = 120.0
    model_api_max_retries: int = 2
    allow_repeat_speaker: bool = False
    consensus: dict[str, Any] = field(default_factory=dict)
    wbft: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "TAMASRunConfig":
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        experiment = payload.get("experiment", {})
        dataset = payload.get("dataset", {})
        agents = payload.get("agents", {})
        collaboration = payload.get("collaboration", {})
        agent_models = dict(agents.get("models", {}))
        capability_coefficients = dict(agents.get("model_capability_coefficients",{}))
        default_model = str(agents.get("default_model", "qwen3.6-35b-a3b"))
        default_capability = agents.get("capability_coefficient")
        proposal_cfg = dict(payload.get("proposal", {}))
        group = dict(payload.get("group", {}))

        return cls(
            config_path=str(config_path),
            output_root=str(experiment.get("output_root", "experiments/exp3_tamas")),
            run_id=str(experiment.get("run_id", "exp3_tamas")),
            dataset_path=dataset.get("path"),
            methods=[str(item) for item in payload.get("methods", [])],
            group_spec=build_group_spec(group),
            default_model=default_model,
            honest_model=str(agent_models.get("honest", default_model)),
            byzantine_model=str(agent_models.get("byzantine", default_model)),
            temperature=float(agents.get("temperature", 0.0)),
            capability_coefficient=float(
                default_capability
                if default_capability is not None
                else capability_coefficients.get(default_model, 1.0)
            ),
            capability_coefficients={
                str(model): float(value)
                for model, value in capability_coefficients.items()
            },
            max_messages=int(collaboration.get("max_messages", 6)),
            timeout_seconds=float(collaboration.get("timeout_seconds", 300.0)),
            request_timeout_seconds=float(collaboration.get("request_timeout_seconds", 120.0)),
            model_api_max_retries=int(collaboration.get("model_api_max_retries", 2)),
            allow_repeat_speaker=bool(collaboration.get("allow_repeat_speaker", False)),
            consensus=proposal_cfg,
            wbft=dict(payload.get("wbft", {})),
        )

    @property
    def model(self) -> str:
        return self.default_model

    @property
    def model_capability_coefficients(self) -> dict[str, float]:
        return self.capability_coefficients

    @property
    def consensus_enabled(self) -> bool:
        return bool(self.group_spec and self.group_spec.uses_consensus)

    @property
    def verification_type(self) -> str:
        return str(self.consensus.get("verification", {}).get("type", "llm"))

    @property
    def include_proposer_as_verifier(self) -> bool:
        return bool(
            self.consensus.get("verification", {}).get(
                "include_proposer_as_verifier",
                False,
            )
        )

    @property
    def dimension_weights(self) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in self.consensus.get("verification", {})
            .get("dimension_weights", {})
            .items()
        }

    @property
    def consensus_strategy(self) -> str:
        return str(self.consensus.get("consensus", {}).get("strategy", "smart_quorum"))

    @property
    def consensus_confidence_threshold(self) -> float:
        return float(self.consensus.get("consensus", {}).get("confidence_threshold", 0.5))

    @property
    def majority_threshold(self) -> float:
        return float(self.consensus.get("consensus", {}).get("majority_threshold", 0.5))

    @property
    def strict_majority(self) -> bool:
        return bool(self.consensus.get("consensus", {}).get("strict_majority", True))

    @property
    def epsilon_ratio(self) -> float:
        return float(self.consensus.get("consensus", {}).get("epsilon_ratio", 0.1))

    @property
    def use_dynamic_estimate(self) -> bool:
        return bool(self.consensus.get("consensus", {}).get("use_dynamic_estimate", False))

    @property
    def max_memory_proposals_per_agent_per_case(self) -> int:
        return int(self.consensus.get("max_proposals_per_agent_per_case", 3))

def build_group_spec(group: dict[str, Any]) -> TamasGroupSpec:
    method = group.get("method", "swarm")
    n = int(group.get("n", 3))
    f = int(group.get("f_value", 0))
    model_regime = group.get("model_regime", "same")
    return TamasGroupSpec(method=method, n=n, f=f, model_regime=model_regime)

