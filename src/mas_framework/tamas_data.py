from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BYZANTINE_DESCRIPTION_KEYWORDS = (
    "unpredictable",
    "fabricated",
)

TAMAS_ATTACK_TYPES = {
    "byzantine",
    "colluding",
    "contradicting",
    "dpi",
    "impersonation",
    "ipi",
}


@dataclass(frozen=True)
class TAMASAgentRole:
    index: int
    agent_name: str
    agent_description: str
    is_byzantine: bool
    role: str
    reason: str


def load_tamas_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = [payload] if isinstance(payload, dict) else list(payload)
    attack_type = infer_attack_type(dataset_path)
    return [
        {
            **case,
            "_tamas_metadata": {
                **dict(case.get("_tamas_metadata", {})),
                "dataset_path": str(dataset_path),
                "attack_type": attack_type,
            },
        }
        for case in cases
    ]


def infer_attack_type(path: str | Path) -> str | None:
    path_obj = Path(path)
    candidates = [path_obj.parent.name, path_obj.stem]
    for candidate in candidates:
        normalized = candidate.lower()
        for attack_type in TAMAS_ATTACK_TYPES:
            if attack_type in normalized:
                return attack_type
    return None


def classify_case_agents(
    case: dict[str, Any],
    *,
    attack_type: str | None = None,
) -> list[TAMASAgentRole]:
    selected_attack_type = attack_type or case.get("_tamas_metadata", {}).get("attack_type")
    return [
        classify_agent(item, index=index, attack_type=selected_attack_type)
        for index, item in enumerate(case.get("agents", []), start=1)
    ]


def classify_agent(
    item: dict[str, Any],
    *,
    index: int = 0,
    attack_type: str | None = None,
) -> TAMASAgentRole:
    is_byzantine, reason = is_byzantine_agent(item, attack_type=attack_type)
    return TAMASAgentRole(
        index=index,
        agent_name=str(item.get("agent_name", "")),
        agent_description=str(item.get("agent_description", "")),
        is_byzantine=is_byzantine,
        role="byzantine" if is_byzantine else "honest",
        reason=reason,
    )


def is_byzantine_agent(
    item: dict[str, Any],
    *,
    attack_type: str | None = None,
) -> tuple[bool, str]:
    normalized_attack = (attack_type or "").lower()
    if normalized_attack in {"", "byzantine"}:
        return _is_byzantine_description(item.get("agent_description", ""))
    return False, f"No Byzantine node rule is configured for attack_type={normalized_attack!r}."


def split_agent_roles(
    roles: Iterable[TAMASAgentRole],
) -> tuple[list[TAMASAgentRole], list[TAMASAgentRole]]:
    role_list = list(roles)
    honest = [role for role in role_list if not role.is_byzantine]
    byzantine = [role for role in role_list if role.is_byzantine]
    return honest, byzantine


def _is_byzantine_description(description: str) -> tuple[bool, str]:
    normalized = str(description).lower()
    matched = [
        keyword
        for keyword in BYZANTINE_DESCRIPTION_KEYWORDS
        if keyword in normalized
    ]
    if matched:
        return True, "Matched Byzantine description keyword(s): " + ", ".join(matched)
    return False, "No Byzantine description keyword matched."


__all__ = [
    "BYZANTINE_DESCRIPTION_KEYWORDS",
    "TAMASAgentRole",
    "classify_agent",
    "classify_case_agents",
    "infer_attack_type",
    "is_byzantine_agent",
    "load_tamas_dataset",
    "split_agent_roles",
]
