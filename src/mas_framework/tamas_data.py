from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_ATTACK_TYPES = {
    "byzantine",
    "colluding",
    "contradicting",
}


def load_tamas_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = [payload] if isinstance(payload, dict) else list(payload)
    attack_type = infer_attack_type(dataset_path)
    if attack_type not in SUPPORTED_ATTACK_TYPES:
        raise NotImplementedError(
            "Only TAMAS-main/data/Byzantine, TAMAS-main/data/Colluding, "
            "and TAMAS-main/data/Contradicting datasets are currently supported. "
            "Other TAMAS attack-type extractors will be implemented later."
        )
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
    candidates = {path_obj.parent.name.lower(), path_obj.stem.lower()}
    if any("byzantine" in candidate for candidate in candidates):
        return "byzantine"
    if any("colluding" in candidate for candidate in candidates):
        return "colluding"
    if any(
        "contradicting" in candidate or "contradinting" in candidate
        for candidate in candidates
    ):
        return "contradicting"
    return None


__all__ = [
    "SUPPORTED_ATTACK_TYPES",
    "infer_attack_type",
    "load_tamas_dataset",
]
