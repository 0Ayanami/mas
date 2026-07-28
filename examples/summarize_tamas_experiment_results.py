from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge TAMAS effectiveness evaluation with local efficiency metrics."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Experiment run directory containing summary.json, cases, traces, and proposals.",
    )
    parser.add_argument(
        "--eval-path",
        default=None,
        help="Path to eval_byzantine.py output. Defaults to <run-dir>/eval_byzantine.json.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Merged result path. Defaults to <run-dir>/experiment_results.json.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    eval_path = Path(args.eval_path) if args.eval_path else run_dir / "eval_byzantine.json"
    output_path = Path(args.output_path) if args.output_path else run_dir / "experiment_results.json"

    result = build_experiment_result(run_dir=run_dir, eval_path=eval_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Results written to: {output_path}")


def build_experiment_result(*, run_dir: Path, eval_path: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json", default={})
    cases = _load_cases(run_dir)
    effectiveness = _read_json(eval_path, default=None)
    efficiency = summary.get("efficiency") or _estimate_efficiency(run_dir, cases)
    return {
        "run_dir": str(run_dir),
        "summary": {
            "run_id": summary.get("run_id", run_dir.name),
            "mode": summary.get("mode"),
            "consensus_enabled": summary.get("consensus_enabled"),
            "consensus_strategy": summary.get("consensus_strategy"),
            "cases": summary.get("cases", len(cases)),
            "effectiveness_source": str(eval_path) if eval_path.exists() else None,
            "efficiency_source": (
                "summary.json"
                if summary.get("efficiency")
                else "estimated_from_saved_traces_and_consensus_artifacts"
            ),
        },
        "effectiveness": (
            effectiveness.get("summary", {}) if isinstance(effectiveness, dict) else None
        ),
        "efficiency": efficiency,
        "consensus": {
            "proposals": summary.get("proposals"),
            "decisions": summary.get("decisions"),
            "accepted_proposals": summary.get("accepted_proposals"),
            "rejected_proposals": summary.get("rejected_proposals"),
            "self_rejected_proposals": summary.get("self_rejected_proposals"),
            "memory_uploaded_proposals": summary.get("memory_uploaded_proposals"),
            "acceptance_rate": summary.get("acceptance_rate"),
            "average_proposal_confidence": summary.get("average_proposal_confidence"),
        },
        "case_efficiency": _case_efficiencies(run_dir, cases),
    }


def _load_cases(run_dir: Path) -> list[dict[str, Any]]:
    cases_json = run_dir / "cases.json"
    if cases_json.exists():
        payload = _read_json(cases_json, default=[])
        if isinstance(payload, list):
            return payload
    cases_dir = run_dir / "cases"
    if not cases_dir.exists():
        return []
    return [
        _read_json(path, default={})
        for path in sorted(cases_dir.glob("*.json"))
    ]


def _estimate_efficiency(run_dir: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_efficiencies = _case_efficiencies(run_dir, cases)
    return {
        "cases": len(case_efficiencies),
        "cases_with_timing": sum(
            1 for item in case_efficiencies if item.get("task_completion_seconds") is not None
        ),
        "total_task_completion_seconds": _sum_present(
            case_efficiencies,
            "task_completion_seconds",
        ),
        "total_consensus_extra_seconds": _sum_present(
            case_efficiencies,
            "consensus_extra_seconds",
        ),
        "total_case_seconds": _sum_present(case_efficiencies, "total_case_seconds"),
        "average_task_completion_seconds": _average_present(
            case_efficiencies,
            "task_completion_seconds",
        ),
        "average_consensus_extra_seconds": _average_present(
            case_efficiencies,
            "consensus_extra_seconds",
        ),
        "average_total_case_seconds": _average_present(
            case_efficiencies,
            "total_case_seconds",
        ),
        "total_interaction_message_count": _sum_present(
            case_efficiencies,
            "interaction_message_count",
        ),
        "total_consensus_extra_message_count": _sum_present(
            case_efficiencies,
            "consensus_extra_message_count",
        ),
        "average_interaction_message_count": _average_present(
            case_efficiencies,
            "interaction_message_count",
        ),
        "average_consensus_extra_message_count": _average_present(
            case_efficiencies,
            "consensus_extra_message_count",
        ),
        "extra_time_cost_ratio": _ratio_float(
            _sum_present(case_efficiencies, "consensus_extra_seconds"),
            _sum_present(case_efficiencies, "task_completion_seconds"),
        ),
        "extra_message_cost_ratio": _ratio_float(
            _sum_present(case_efficiencies, "consensus_extra_message_count"),
            _sum_present(case_efficiencies, "interaction_message_count"),
        ),
    }


def _case_efficiencies(run_dir: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions = _read_json(run_dir / "consensus_decisions.json", default=[])
    if not isinstance(decisions, list):
        decisions = []
    result = []
    for case in cases:
        case_id = case.get("case_id")
        existing = dict(case.get("efficiency", {}) or {})
        if not case_id:
            continue
        if not existing:
            existing = {
                "task_completion_seconds": None,
                "consensus_extra_seconds": None,
                "total_case_seconds": None,
                "interaction_message_count": _trace_message_count(run_dir, case_id),
                "consensus_extra_message_count": _estimated_consensus_extra_messages(
                    case=case,
                    decisions=decisions,
                ),
                "extra_time_cost_ratio": None,
                "extra_message_cost_ratio": None,
            }
            existing["extra_message_cost_ratio"] = _ratio_float(
                existing["consensus_extra_message_count"],
                existing["interaction_message_count"],
            )
        result.append({"case_id": case_id, **existing})
    return result


def _trace_message_count(run_dir: Path, case_id: str) -> int | None:
    trace_path = run_dir / "traces" / f"{case_id}.txt"
    if not trace_path.exists():
        return None
    trace = trace_path.read_text(encoding="utf-8")
    if not trace.strip():
        return 0
    return len(
        [
            block
            for block in re.split(r"\n\s*\n", trace)
            if re.match(r"^[A-Za-z0-9_ -]+:", block.strip())
        ]
    )


def _estimated_consensus_extra_messages(
    *,
    case: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> int:
    case_id = case.get("case_id")
    proposal_count = int(case.get("proposal_count", 0) or 0)
    verifier_votes = 0
    for decision in decisions:
        experiment = decision.get("_experiment", {})
        if experiment.get("case_id") != case_id:
            continue
        verifier_votes += len(decision.get("votes", []) or [])
    return proposal_count + verifier_votes


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_present(items: list[dict[str, Any]], key: str) -> float | int | None:
    values = [item.get(key) for item in items if item.get(key) is not None]
    if not values:
        return None
    return sum(values)


def _average_present(items: list[dict[str, Any]], key: str) -> float | None:
    values = [item.get(key) for item in items if item.get(key) is not None]
    if not values:
        return None
    return float(sum(values)) / len(values)


def _ratio_float(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    main()
