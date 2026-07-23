from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Literal
from pathlib import Path
import yaml
import json

Exp1Method = Literal[
    "discussion_based",
    "discussion_based_cp_wbft",
    "discussion_based_consensus",
]
ModelRegime = Literal[
    "same",
    "weak_byzantine_strong_honest",
    "strong_byzantine_weak_honest",
]



@dataclass(frozen=True)
class MMLUProQuestion:
    question_id: str
    category: str
    question: str
    options: list[str]
    answer: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Exp1GroupSpec:
    method: Exp1Method
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
class Exp1Config:
    config_path: str
    output_root: str = "experiments/exp1_mmlu_pro"
    run_id: str = "exp1_mmlu_pro"
    seed: int = 42
    dataset_path: str | None = None
    category_count: int = 13
    samples_per_category: int = 3
    methods: list[str] = field(default_factory=list)
    group_specs: list[Exp1GroupSpec] = field(default_factory=list)
    default_model: str = "qwen3.6-35b-a3b"
    honest_model: str = "qwen3.6-35b-a3b"
    byzantine_model: str = "qwen3.6-35b-a3b"
    temperature: float = 0.0
    capability_coefficients: dict[str, float] = field(default_factory=dict)
    max_messages: int = 6
    timeout_seconds: float = 300.0
    request_timeout_seconds: float = 120.0
    model_api_max_retries: int = 2
    allow_repeat_speaker: bool = False
    consensus: dict[str, Any] = field(default_factory=dict)
    wbft: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Exp1Config":
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
        group_specs = build_group_matrix(payload)
        return cls(
            config_path=str(config_path),
            output_root=str(experiment.get("output_root", "experiments/exp1_mmlu_pro")),
            run_id=str(experiment.get("run_id", "exp1_mmlu_pro")),
            seed=int(experiment.get("seed", 42)),
            dataset_path=dataset.get("path"),
            category_count=int(dataset.get("categories_per_sample", 13)),
            samples_per_category=int(dataset.get("samples_per_category", 3)),
            methods=[str(item) for item in payload.get("methods", [])],
            group_specs=group_specs,
            default_model=default_model,
            honest_model=str(agent_models.get("honest", default_model)),
            byzantine_model=str(agent_models.get("byzantine", default_model)),
            temperature=float(agents.get("temperature", 0.0)),
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


@dataclass(frozen=True)
class Exp1AgentSpec:
    agent_id: str
    display_name: str
    model: str
    is_byzantine: bool
    capability_coefficient: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Exp1CaseResult:
    case_id: str
    question: MMLUProQuestion
    predicted_answer: str | None
    correct: bool
    final_text: str
    trace: str
    metrics: dict[str, Any]
    agent_specs: dict[str, dict[str, Any]]
    consensus_decisions: list[dict[str, Any]] = field(default_factory=list)
    memory_proposals: list[dict[str, Any]] = field(default_factory=list)
    shared_task_memory: list[dict[str, Any]] = field(default_factory=list)
    wbft_result: dict[str, Any] | None = None

    def to_case_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question.to_dict(),
            "predicted_answer": self.predicted_answer,
            "correct_answer": self.question.answer,
            "correct": self.correct,
            "final_text": self.final_text,
            "metrics": self.metrics,
            "agent_specs": self.agent_specs,
            "proposal_count": len(self.memory_proposals),
            "shared_task_memory": self.shared_task_memory,
            "shared_task_memory_count": len(self.shared_task_memory),
            "decision_count": len(self.consensus_decisions),
            "wbft_result": self.wbft_result,
        }

@dataclass
class TaskMemoryPool:
    """Ephemeral, consensus-gated memory shared only while one case is running."""

    task_id: str
    accepted_proposals: list[Any] = field(default_factory=list)

    def add(self, proposal: Any) -> None:
        self.accepted_proposals.append(proposal)

    def payloads(self) -> list[dict[str, Any]]:
        return [proposal.to_dict() for proposal in self.accepted_proposals]

    def coordinator_message(self, *, decision: str) -> str:
        if not self.accepted_proposals:
            return (
                "Memory coordinator: the submitted proposal was not accepted."
                "Do not use it as shared evidence; continue the group discussion."
            )
        entries = self.payloads()
        return (
            "Memory coordinator: consensus accepted the latest proposal "
            f"(decision={decision}). The following is the task-scoped shared memory. "
            "Treat it as verified context, cite it by proposal_id when useful, and continue solving.\n"
            f"SHARED_TASK_MEMORY\n{json.dumps(entries, ensure_ascii=False)}\nEND_SHARED_TASK_MEMORY"
        )


def build_group_matrix(config_payload: dict[str, Any]) -> list[Exp1GroupSpec]:
    methods = [str(item) for item in config_payload.get("methods", [])]
    specs: list[Exp1GroupSpec] = []
    for group in config_payload.get("groups", []):
        n = int(group["n"])
        for f_value in group.get("f_values", []):
            for regime in group.get("model_regimes", ["same"]):
                for method in methods:
                    specs.append(
                        Exp1GroupSpec(
                            method=method,
                            n=n,
                            f=int(f_value),
                            model_regime=str(regime),
                        )
                    )
    return specs