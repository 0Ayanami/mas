from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mas_framework.consensus import VerificationVector
from mas_framework.exp3.tamas_workflow import TAMASAutoGenRunner, TAMASRunConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TAMAS Byzantine experiments with shared memory and consensus tracing."
    )
    parser.add_argument(
        "--data-dir",
        default="TAMAS-main/data/Byzantine",
        help="Directory containing TAMAS Byzantine JSON files.",
    )
    parser.add_argument(
        "--config",
        default="src/mas_framework/configs/experiment_configs/unified_config.yaml",
        help="Unified experiment config YAML.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Experiment output directory. Defaults to config experiment.output_root.",
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Optional maximum cases to run per dataset file.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Case index to start from within each dataset file. Existing lower-index outputs are reused.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Use an existing run id instead of creating a timestamped one.",
    )
    parser.add_argument(
        "--consensus",
        choices=["config", "enabled", "disabled"],
        default="config",
        help="Override proposal consensus. Defaults to the YAML config value.",
    )
    parser.add_argument(
        "--shared-memory-across-cases",
        action="store_true",
        help="Deprecated compatibility flag. Consensus runs now share one Mem0 user_id per experiment group by default.",
    )
    parser.add_argument(
        "--separate-memory-per-case",
        action="store_true",
        help="Use a separate Mem0 user_id for each case instead of one shared user_id for the experiment group.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = TAMASRunConfig.from_unified_config(config_path)
    if args.consensus == "enabled":
        config = replace(config, consensus_enabled=True)
    elif args.consensus == "disabled":
        config = replace(config, consensus_enabled=False)

    payload = _read_yaml(config_path)
    experiment_cfg = payload.get("experiment", {})
    run_id = args.run_id or _run_id(experiment_cfg.get("run_id"))
    output_root = Path(args.output_root or experiment_cfg.get("output_root", "experiments/TAMAS"))
    data_path = Path(args.data_dir)
    dataset_scope = _dataset_scope(data_path)
    output_dir = output_root / dataset_scope / run_id
    cases_dir = output_dir / "cases"
    proposals_dir = output_dir / "memory_proposals"
    traces_dir = output_dir / "traces"
    for directory in (cases_dir, proposals_dir, traces_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_files = [data_path] if data_path.is_file() else sorted(data_path.glob("*.json"))
    all_case_summaries: list[dict[str, Any]] = []
    all_proposals: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    experiment_memory_user_id = _experiment_memory_user_id(
        base_user_id=config.memory_user_id,
        dataset_scope=dataset_scope,
        consensus_enabled=config.consensus_enabled,
        consensus_strategy=config.consensus_strategy,
        run_id=run_id,
    )
    if config.consensus_enabled:
        config = replace(config, memory_user_id=experiment_memory_user_id)

    for dataset_file in dataset_files:
        domain = _domain_name(dataset_file)
        cases = TAMASAutoGenRunner.load_dataset(dataset_file)
        if args.start_index:
            _load_existing_outputs(
                domain=domain,
                start_index=args.start_index,
                cases_dir=cases_dir,
                proposals_dir=proposals_dir,
                all_case_summaries=all_case_summaries,
                all_proposals=all_proposals,
                all_decisions=all_decisions,
            )
        if args.limit_per_file is not None:
            cases = cases[: args.limit_per_file]

        for case_index, case in enumerate(cases):
            if case_index < args.start_index:
                continue
            case_id = f"{domain}_{case_index:03d}"
            case_config = config
            if config.consensus_enabled and args.separate_memory_per_case:
                case_config = replace(config, memory_user_id=f"{config.memory_user_id}:{run_id}:{case_id}")

            runner = TAMASAutoGenRunner(config=case_config)
            result = runner.run_case(case, task_id=case_id)
            agent_specs = {
                agent_id: _plain(spec)
                for agent_id, spec in sorted(runner._agent_specs.items())
            }
            decision_payloads = [
                _decision_payload(
                    decision=decision,
                    domain=domain,
                    case_index=case_index,
                    case_id=case_id,
                )
                for decision in result.consensus_decisions
            ]
            proposal_payloads = [
                _proposal_payload(
                    proposal=proposal,
                    domain=domain,
                    case_index=case_index,
                    case_id=case_id,
                    memory_user_id=case_config.memory_user_id,
                    dimension_weights=case_config.dimension_weights,
                    decisions=decision_payloads,
                )
                for proposal in result.memory_proposals
            ]
            case_summary = _case_summary(
                domain=domain,
                case_index=case_index,
                case_id=case_id,
                case=case,
                result=result,
                agent_specs=agent_specs,
                proposal_payloads=proposal_payloads,
                decision_payloads=decision_payloads,
                memory_user_id=case_config.memory_user_id,
                config=case_config,
            )
            all_case_summaries.append(case_summary)
            all_proposals.extend(proposal_payloads)
            all_decisions.extend(decision_payloads)

            _write_json(cases_dir / f"{case_id}.json", case_summary)
            _write_json(proposals_dir / f"{case_id}.json", proposal_payloads)
            (traces_dir / f"{case_id}.txt").write_text(result.trace, encoding="utf-8")
            print(
                f"[{case_id}] proposals={len(proposal_payloads)} "
                f"accepted={case_summary['accepted_proposals']} "
                f"rejected={case_summary['rejected_proposals']}"
            )

    summary = _experiment_summary(
        run_id=run_id,
        config_path=config_path,
        data_dir=Path(args.data_dir),
        memory_user_id=config.memory_user_id if config.consensus_enabled else None,
        case_summaries=all_case_summaries,
        proposals=all_proposals,
        decisions=all_decisions,
        config=config,
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "cases.json", all_case_summaries)
    _write_json(output_dir / "memory_proposals.json", all_proposals)
    _write_json(output_dir / "consensus_decisions.json", all_decisions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results written to: {output_dir}")


def _proposal_payload(
    *,
    proposal: Any,
    domain: str,
    case_index: int,
    case_id: str,
    memory_user_id: str,
    dimension_weights: dict[str, float],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    data = proposal.to_dict() if hasattr(proposal, "to_dict") else _plain(proposal)
    proposal_id = data.get("header", {}).get("proposal_id")
    decision = next(
        (
            item
            for item in decisions
            if item.get("proposal_id") == proposal_id
        ),
        None,
    )
    lifecycle = _proposal_lifecycle(decision)
    data["_experiment"] = {
        "domain": domain,
        "case_index": case_index,
        "case_id": case_id,
        "memory_user_id": memory_user_id,
        "proposal_lifecycle": lifecycle["proposal_lifecycle"],
        "consensus_result": lifecycle["consensus_result"],
        "memory_uploaded": lifecycle["memory_uploaded"],
        "self_confidence_score": _self_confidence_score(
            proposal,
            dimension_weights=dimension_weights,
        ),
    }
    return data


def _load_existing_outputs(
    *,
    domain: str,
    start_index: int,
    cases_dir: Path,
    proposals_dir: Path,
    all_case_summaries: list[dict[str, Any]],
    all_proposals: list[dict[str, Any]],
    all_decisions: list[dict[str, Any]],
) -> None:
    for case_index in range(start_index):
        case_id = f"{domain}_{case_index:03d}"
        case_path = cases_dir / f"{case_id}.json"
        proposals_path = proposals_dir / f"{case_id}.json"
        if case_path.exists():
            all_case_summaries.append(json.loads(case_path.read_text(encoding="utf-8")))
        if proposals_path.exists():
            proposal_payloads = json.loads(proposals_path.read_text(encoding="utf-8"))
            all_proposals.extend(proposal_payloads)
            all_decisions.extend(
                _decision_payload_from_proposal(
                    proposal=proposal,
                    domain=domain,
                    case_index=case_index,
                    case_id=case_id,
                )
                for proposal in proposal_payloads
                if proposal.get("verification", {}).get("consensus_result") is not None
            )


def _decision_payload_from_proposal(
    *,
    proposal: dict[str, Any],
    domain: str,
    case_index: int,
    case_id: str,
) -> dict[str, Any]:
    decision = dict(proposal.get("verification", {}).get("consensus_result") or {})
    decision.setdefault("proposal_id", proposal.get("header", {}).get("proposal_id"))
    experiment = proposal.get("_experiment", {})
    if "metadata" not in decision:
        confidence = None
        if decision.get("result") == "pass":
            confidence = experiment.get("self_confidence_score")
        decision["metadata"] = {"proposal_confidence_score": confidence or 0.0}
    decision["_experiment"] = {
        "domain": domain,
        "case_index": case_index,
        "case_id": case_id,
    }
    return decision


def _proposal_lifecycle(decision: dict[str, Any] | None) -> dict[str, Any]:
    if decision is None:
        return {
            "proposal_lifecycle": "self_rejected",
            "consensus_result": None,
            "memory_uploaded": False,
        }
    result = decision.get("result")
    if result == "pass":
        lifecycle = "consensus_accepted"
        memory_uploaded = True
    elif result == "pending":
        lifecycle = "consensus_pending"
        memory_uploaded = False
    else:
        lifecycle = "consensus_rejected"
        memory_uploaded = False
    return {
        "proposal_lifecycle": lifecycle,
        "consensus_result": result,
        "memory_uploaded": memory_uploaded,
    }


def _self_confidence_score(
    proposal: Any,
    *,
    dimension_weights: dict[str, float],
) -> float | None:
    self_verification = getattr(getattr(proposal, "verification", None), "self_verification", None)
    if self_verification is None:
        return None
    kwargs: dict[str, Any] = {
        "veracity": int(round(float(self_verification.veracity_score))),
        "rationality": int(round(float(self_verification.rationality_score))),
        "value": int(round(float(self_verification.value_score))),
        "security": int(round(float(self_verification.security_score))),
        "verifier_agent_id": getattr(proposal, "agent_id", None),
        "metadata": {"source": "self_verification"},
    }
    if dimension_weights:
        kwargs["dimension_weights"] = dimension_weights
    return VerificationVector(**kwargs).confidence_score


def _decision_payload(
    *,
    decision: Any,
    domain: str,
    case_index: int,
    case_id: str,
) -> dict[str, Any]:
    data = decision.to_dict() if hasattr(decision, "to_dict") else _plain(decision)
    data["_experiment"] = {
        "domain": domain,
        "case_index": case_index,
        "case_id": case_id,
    }
    return data


def _case_summary(
    *,
    domain: str,
    case_index: int,
    case_id: str,
    case: dict[str, Any],
    result: Any,
    agent_specs: dict[str, Any],
    proposal_payloads: list[dict[str, Any]],
    decision_payloads: list[dict[str, Any]],
    memory_user_id: str,
    config: TAMASRunConfig,
) -> dict[str, Any]:
    accepted = [item for item in decision_payloads if item.get("result") == "pass"]
    rejected = [item for item in decision_payloads if item.get("result") == "fail"]
    pending = [item for item in decision_payloads if item.get("result") == "pending"]
    memory_uploaded = [
        item for item in proposal_payloads if item.get("_experiment", {}).get("memory_uploaded")
    ]
    byzantine_agent_ids = {
        agent_id for agent_id, spec in agent_specs.items() if spec.get("is_byzantine")
    }
    byzantine_proposals = [
        item for item in proposal_payloads if item.get("header", {}).get("agent_id") in byzantine_agent_ids
    ]
    accepted_ids = {item.get("proposal_id") for item in accepted}
    accepted_byzantine = [
        item for item in byzantine_proposals if item.get("header", {}).get("proposal_id") in accepted_ids
    ]
    return {
        "domain": domain,
        "case_index": case_index,
        "case_id": case_id,
        "memory_user_id": memory_user_id,
        "consensus_enabled": config.consensus_enabled,
        "consensus_strategy": config.consensus_strategy if config.consensus_enabled else None,
        "user_query": case.get("user query", ""),
        "agent_specs": agent_specs,
        "final_text": result.final_text,
        "efficiency": _case_efficiency(result),
        "proposal_count": len(proposal_payloads),
        "decision_count": len(decision_payloads),
        "self_rejected_proposals": max(len(proposal_payloads) - len(decision_payloads), 0),
        "accepted_proposals": len(accepted),
        "rejected_proposals": len(rejected),
        "pending_proposals": len(pending),
        "memory_uploaded_proposals": len(memory_uploaded),
        "byzantine_proposals": len(byzantine_proposals),
        "accepted_byzantine_proposals": len(accepted_byzantine),
        "acceptance_rate": _ratio(len(accepted), len(decision_payloads)),
        "byzantine_acceptance_rate": _ratio(len(accepted_byzantine), len(byzantine_proposals)),
        "proposal_ids": [item.get("header", {}).get("proposal_id") for item in proposal_payloads],
    }


def _experiment_summary(
    *,
    run_id: str,
    config_path: Path,
    data_dir: Path,
    memory_user_id: str | None,
    case_summaries: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    config: TAMASRunConfig,
) -> dict[str, Any]:
    accepted = [item for item in decisions if item.get("result") == "pass"]
    rejected = [item for item in decisions if item.get("result") == "fail"]
    pending = [item for item in decisions if item.get("result") == "pending"]
    by_domain: dict[str, dict[str, Any]] = {}
    for case in case_summaries:
        bucket = by_domain.setdefault(
            case["domain"],
            {
                "cases": 0,
                "proposals": 0,
                "accepted_proposals": 0,
                "rejected_proposals": 0,
                "pending_proposals": 0,
                "self_rejected_proposals": 0,
                "memory_uploaded_proposals": 0,
                "byzantine_proposals": 0,
                "accepted_byzantine_proposals": 0,
            },
        )
        bucket["cases"] += 1
        bucket["proposals"] += case["proposal_count"]
        bucket["accepted_proposals"] += case["accepted_proposals"]
        bucket["rejected_proposals"] += case["rejected_proposals"]
        bucket["pending_proposals"] += case["pending_proposals"]
        bucket["self_rejected_proposals"] += case["self_rejected_proposals"]
        bucket["memory_uploaded_proposals"] += case["memory_uploaded_proposals"]
        bucket["byzantine_proposals"] += case["byzantine_proposals"]
        bucket["accepted_byzantine_proposals"] += case["accepted_byzantine_proposals"]
    for bucket in by_domain.values():
        bucket["acceptance_rate"] = _ratio(bucket["accepted_proposals"], bucket["proposals"])
        bucket["byzantine_acceptance_rate"] = _ratio(
            bucket["accepted_byzantine_proposals"],
            bucket["byzantine_proposals"],
        )
    confidence_scores = [
        float(item.get("metadata", {}).get("proposal_confidence_score", 0.0))
        for item in decisions
    ]
    efficiency = _experiment_efficiency(case_summaries)
    return {
        "run_id": run_id,
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "consensus_enabled": config.consensus_enabled,
        "consensus_strategy": config.consensus_strategy if config.consensus_enabled else None,
        "memory_user_id": memory_user_id,
        "cases": len(case_summaries),
        "proposals": len(proposals),
        "decisions": len(decisions),
        "accepted_proposals": len(accepted),
        "rejected_proposals": len(rejected),
        "pending_proposals": len(pending),
        "self_rejected_proposals": sum(
            case.get("self_rejected_proposals", 0) for case in case_summaries
        ),
        "memory_uploaded_proposals": sum(
            case.get("memory_uploaded_proposals", 0) for case in case_summaries
        ),
        "acceptance_rate": _ratio(len(accepted), len(decisions)),
        "average_proposal_confidence": (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        ),
        "efficiency": efficiency,
        "by_domain": by_domain,
    }


def _case_efficiency(result: Any) -> dict[str, Any]:
    metrics = dict(getattr(result, "run_metrics", {}) or {})
    task_seconds = metrics.get("task_completion_seconds")
    extra_seconds = metrics.get("consensus_extra_seconds")
    task_messages = metrics.get("interaction_message_count")
    extra_messages = metrics.get("consensus_extra_message_count")
    return {
        "task_completion_seconds": task_seconds,
        "consensus_extra_seconds": extra_seconds,
        "total_case_seconds": metrics.get("total_case_seconds"),
        "interaction_message_count": task_messages,
        "consensus_extra_message_count": extra_messages,
        "extra_time_cost_ratio": _ratio_float(extra_seconds, task_seconds),
        "extra_message_cost_ratio": _ratio_float(extra_messages, task_messages),
    }


def _experiment_efficiency(case_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    efficiencies = [case.get("efficiency", {}) for case in case_summaries]
    total_task_seconds = _sum_present(efficiencies, "task_completion_seconds")
    total_extra_seconds = _sum_present(efficiencies, "consensus_extra_seconds")
    total_case_seconds = _sum_present(efficiencies, "total_case_seconds")
    total_task_messages = _sum_present(efficiencies, "interaction_message_count")
    total_extra_messages = _sum_present(efficiencies, "consensus_extra_message_count")
    cases_with_timing = sum(
        1
        for item in efficiencies
        if item.get("task_completion_seconds") is not None
    )
    return {
        "cases": len(case_summaries),
        "cases_with_timing": cases_with_timing,
        "total_task_completion_seconds": total_task_seconds,
        "total_consensus_extra_seconds": total_extra_seconds,
        "total_case_seconds": total_case_seconds,
        "average_task_completion_seconds": _average_present(
            efficiencies,
            "task_completion_seconds",
        ),
        "average_consensus_extra_seconds": _average_present(
            efficiencies,
            "consensus_extra_seconds",
        ),
        "average_total_case_seconds": _average_present(
            efficiencies,
            "total_case_seconds",
        ),
        "total_interaction_message_count": total_task_messages,
        "total_consensus_extra_message_count": total_extra_messages,
        "average_interaction_message_count": _average_present(
            efficiencies,
            "interaction_message_count",
        ),
        "average_consensus_extra_message_count": _average_present(
            efficiencies,
            "consensus_extra_message_count",
        ),
        "extra_time_cost_ratio": _ratio_float(total_extra_seconds, total_task_seconds),
        "extra_message_cost_ratio": _ratio_float(total_extra_messages, total_task_messages),
    }


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


def _run_id(config_run_id: Any) -> str:
    prefix = str(config_run_id or "run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def _dataset_scope(data_path: Path) -> str:
    source = data_path.parent.name if data_path.is_file() else data_path.name
    return _slug(source or "dataset")


def _domain_name(dataset_file: Path) -> str:
    name = dataset_file.stem
    for suffix in ("_byzantine", "_colluding"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _experiment_memory_user_id(
    *,
    base_user_id: str,
    dataset_scope: str,
    consensus_enabled: bool,
    consensus_strategy: str,
    run_id: str,
) -> str:
    consensus_scope = consensus_strategy if consensus_enabled else "no_consensus"
    return ":".join(
        _slug(part)
        for part in (
            base_user_id,
            dataset_scope,
            consensus_scope,
            run_id,
        )
        if part
    )


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    normalized = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in normalized.split("_") if part) or "default"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {key: _plain(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
