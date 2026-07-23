from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Literal
from pathlib import Path
import yaml
import json
import re

DEFAULT_MMLU_PRO_SAMPLE_PATH = Path("src/mas_framework/mmlu/mmlu_pro_13x3_seed42.json")

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
    @property
    def task(self) -> str:
        return format_qa_task(self.question, self.options)


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
    group_spec: Exp1GroupSpec
    output_root: str = "experiments/exp1_mmlu_pro"
    run_id: str = "exp1_mmlu_pro"
    seed: int = 42
    dataset_path: str | None = None
    category_count: int = 13
    samples_per_category: int = 3
    methods: list[str] = field(default_factory=list)
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
        group = dict(payload.get("group", {}))
        return cls(
            config_path=str(config_path),
            output_root=str(experiment.get("output_root", "experiments/exp1_mmlu_pro")),
            run_id=str(experiment.get("run_id", "exp1_mmlu_pro")),
            seed=int(experiment.get("seed", 42)),
            dataset_path=dataset.get("path"),
            category_count=int(dataset.get("categories_per_sample", 13)),
            samples_per_category=int(dataset.get("samples_per_category", 3)),
            methods=[str(item) for item in payload.get("methods", [])],
            group_spec=build_group_spec(group),
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
    answer_stats: dict[str, Any] = field(default_factory=dict)
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
            "answer_stats": self.answer_stats,
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

    def coordinator_message(self) -> str:
        if not self.accepted_proposals:
            return (
                "Memory coordinator: the submitted proposal was not accepted."
                "Do not use it as shared evidence; continue the group discussion."
            )
        entries = self.payloads()
        return (
            "Memory coordinator: the following task-scoped memory proposals "
            "have passed consensus and may be used as shared context for this task only. "
            "Treat it as verified context, cite it by proposal_id when useful, and continue solving.\n"
            f"SHARED_TASK_MEMORY\n{json.dumps(entries, ensure_ascii=False)}\nEND_SHARED_TASK_MEMORY"
        )

def load_mmlu_pro_questions(
    path: str | Path = DEFAULT_MMLU_PRO_SAMPLE_PATH,
) -> list[MMLUProQuestion]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"MMLU-Pro sample path does not exist: {dataset_path}. "
            "Use src/mas_framework/mmlu/mmlu_pro_13x3_seed42.json "
            "or pass --dataset-path explicitly."
    )
    records = _read_dataset_records(dataset_path)
    return [_normalize_mmlu_record(record, index) for index, record in enumerate(records)]


def build_group_spec(group: dict[str, Any]) -> Exp1GroupSpec:
    method = group.get("method", "discussion_based")
    n = int(group.get("n", 3))
    f = int(group.get("f_value", 0))
    model_regime = group.get("model_regime", "same")
    return Exp1GroupSpec(method=method, n=n, f=f, model_regime=model_regime)

def format_qa_task(question: str, options: list[str]) -> str:
    opts = "\n".join(
        f"{chr(ord('A') + index)}. {option}"
        for index, option in enumerate(options)
    )
    return (
        "Answer the following multiple-choice question.\n"
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
        "Return the final answer exactly as ANSWER:(<LETTER>)."
    )

def _read_dataset_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    records = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"No MMLU-Pro records found in {path}.")
    return records


def _normalize_mmlu_record(record: dict[str, Any], index: int) -> MMLUProQuestion:
    question_id = str(record["question_id"])
    category = str(record["category"])
    question = str(record["question"])
    options_value = record["options"]
    if not isinstance(options_value, list):
        raise ValueError(f"Cannot normalize MMLU-Pro record at index {index}: {record}")
    options = [str(option) for option in options_value]
    answer = str(record["answer"]).strip().upper()
    if not question or not options or not re.fullmatch(r"[A-J]", answer):
        raise ValueError(f"Cannot normalize MMLU-Pro record at index {index}: {record}")
    return MMLUProQuestion(
        question_id=_slug(question_id),
        category=category,
        question=question,
        options=options,
        answer=answer,
        raw=dict(record["raw"]),
    )

def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    normalized = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in normalized.split("_") if part) or "item"

def _model_result_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item) for item in content)
    return str(content)

def _loads_json_object(candidate: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        repaired = repaired.replace('""data"', '"data"')
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None

def _extract_json_object(content: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    candidate = (
        fenced.group(1)
        if fenced
        else content[content.find("{") : content.rfind("}") + 1]
    )
    if not candidate or not candidate.startswith("{"):
        return None
    payload = _loads_json_object(candidate)
    if payload is None:
        return None
    return payload if isinstance(payload, dict) else None

