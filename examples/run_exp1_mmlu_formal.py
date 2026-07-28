from __future__ import annotations

import argparse
import asyncio
import csv
import json
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mas_framework.exp_1.exp1_mmlu import Exp1Runner, summarize_group_results
from mas_framework.exp_1.exp1_mmlu_crewai import CrewAIMMLURunner
from mas_framework.exp_1.models import (
    Exp1CaseResult,
    Exp1Config,
    Exp1GroupSpec,
    MMLUProQuestion,
    load_mmlu_pro_questions,
)


AUTOGEN_METHODS = (
    "discussion_based",
    "discussion_based_consensus",
    "discussion_based_cp_wbft",
)
CREWAI_METHODS = (
    "role_based",
    "role_based_consensus",
    "role_based_cp_wbft",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run formal Exp1 MMLU experiments over AutoGen and CrewAI workflows."
    )
    parser.add_argument(
        "--autogen-config",
        default="src/mas_framework/configs/experiment_configs/exp1_mmlu_pro.yaml",
    )
    parser.add_argument(
        "--crewai-config",
        default="src/mas_framework/configs/experiment_configs/exp1_mmlu_crewai.yaml",
    )
    parser.add_argument(
        "--dataset-path",
        default="src/mas_framework/mmlu/mmlu_pro_13x3_seed42.json",
    )
    parser.add_argument("--output-root", default="experiments/exp1_mmlu_pro")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--framework", choices=["all", "autogen", "crewai"], default="all")
    parser.add_argument("--method", default=None)
    parser.add_argument("--f-values", default="0,1,2")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case-retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild all_results.json/csv from existing group summaries without API calls.",
    )
    args = parser.parse_args()

    run_id = args.run_id or f"formal_mmlu_n3_same_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        summaries = _load_existing_summaries(run_dir)
        _write_json(run_dir / "all_results.json", summaries)
        _write_csv(run_dir / "all_results.csv", summaries)
        print(f"Aggregated {len(summaries)} summaries into: {run_dir}")
        return

    questions = load_mmlu_pro_questions(args.dataset_path)
    if args.limit_cases is not None:
        questions = questions[: args.limit_cases]
    _write_json(run_dir / "dataset_sample.json", [question.to_dict() for question in questions])

    autogen_config = Exp1Config.from_yaml(args.autogen_config)
    crewai_config = Exp1Config.from_yaml(args.crewai_config)
    shutil.copyfile(args.autogen_config, run_dir / "autogen_config_snapshot.yaml")
    shutil.copyfile(args.crewai_config, run_dir / "crewai_config_snapshot.yaml")

    groups = _build_formal_groups(
        framework=args.framework,
        method=args.method,
        f_values=_csv_ints(args.f_values),
    )
    plan = {
        "run_dir": str(run_dir),
        "dataset_path": args.dataset_path,
        "case_count": len(questions),
        "groups": [group_plan for group_plan in groups],
    }
    _write_json(run_dir / "experiment_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    all_summaries: list[dict[str, Any]] = []
    for group_plan in groups:
        framework = group_plan["framework"]
        group = Exp1GroupSpec(
            method=group_plan["method"],
            n=3,
            f=int(group_plan["f"]),
            model_regime="same",
        )
        base_config = autogen_config if framework == "autogen" else crewai_config
        runner_cls = Exp1Runner if framework == "autogen" else CrewAIMMLURunner
        group_dir = run_dir / "groups" / _group_dir_name(framework, group)
        summary_path = group_dir / "summary.json"
        if args.resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            config = replace(
                base_config,
                group_spec=group,
                output_root=str(group_dir),
                run_id=".",
                dataset_path=args.dataset_path,
                category_count=5,
                samples_per_category=1,
            )
            summary = _run_group(
                framework=framework,
                runner=runner_cls(config=config),
                group=group,
                questions=questions,
                group_dir=group_dir,
                case_retries=args.case_retries,
                retry_sleep=args.retry_sleep,
            )
        all_summaries.append(summary)
        _write_json(run_dir / "all_results.json", all_summaries)
        _write_csv(run_dir / "all_results.csv", all_summaries)
        print(
            f"[{summary['framework']}::{group.group_id}] "
            f"accuracy={summary['accuracy']:.4f} cases={summary['cases']} "
            f"time={summary['time'].get('total_seconds', 0.0):.2f}s"
        )

    _write_json(run_dir / "all_results.json", all_summaries)
    _write_csv(run_dir / "all_results.csv", all_summaries)
    print(f"Results written to: {run_dir}")


def _load_existing_summaries(run_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((run_dir / "groups").glob("*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def _build_formal_groups(
    *,
    framework: str,
    method: str | None,
    f_values: Iterable[int],
) -> list[dict[str, Any]]:
    frameworks = ["autogen", "crewai"] if framework == "all" else [framework]
    groups: list[dict[str, Any]] = []
    for current_framework in frameworks:
        methods = AUTOGEN_METHODS if current_framework == "autogen" else CREWAI_METHODS
        for current_method in methods:
            if method is not None and current_method != method:
                continue
            for f in f_values:
                groups.append(
                    {
                        "framework": current_framework,
                        "method": current_method,
                        "n": 3,
                        "f": int(f),
                        "model_regime": "same",
                    }
                )
    return groups


def _run_group(
    *,
    framework: str,
    runner: Any,
    group: Exp1GroupSpec,
    questions: list[MMLUProQuestion],
    group_dir: Path,
    case_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    cases_dir = group_dir / "cases"
    traces_dir = group_dir / "traces"
    proposals_dir = group_dir / "memory_proposals"
    errors_dir = group_dir / "errors"
    for directory in (cases_dir, traces_dir, proposals_dir, errors_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_json(
        group_dir / "group_config.json",
        {
            "framework": framework,
            "group": group.to_dict(),
            "memory_scope": "task_scoped_consensus_memory" if group.uses_consensus else None,
            "external_mem0": False,
        },
    )

    results = _run_cases(
        framework=framework,
        runner=runner,
        group=group,
        questions=questions,
        cases_dir=cases_dir,
        traces_dir=traces_dir,
        proposals_dir=proposals_dir,
        errors_dir=errors_dir,
        case_retries=case_retries,
        retry_sleep=retry_sleep,
    )
    all_decisions: list[dict[str, Any]] = []
    wbft_results: list[dict[str, Any]] = []
    for result in results:
        all_decisions.extend(result.consensus_decisions)
        if result.wbft_result is not None:
            wbft_results.append({"case_id": result.case_id, **result.wbft_result})

    finish_group = getattr(runner, "finish_group", None)
    if callable(finish_group):
        finish_group()

    summary = summarize_group_results(group, results)
    summary["framework"] = framework
    summary["group_dir"] = str(group_dir)
    summary["memory_proposal_metrics"] = _memory_proposal_metrics(results)
    summary["consensus_vote_metrics"] = _consensus_vote_metrics(all_decisions)
    summary["agent_weight_metrics"] = _agent_weight_metrics(all_decisions, group_dir)
    summary["wbft_metrics"] = _wbft_metrics(wbft_results)

    _write_json(group_dir / "summary.json", summary)
    _write_json(group_dir / "regular_metrics.json", summary["regular_metrics"])
    _write_json(group_dir / "memory_proposal_metrics.json", summary["memory_proposal_metrics"])
    _write_json(group_dir / "consensus_vote_metrics.json", summary["consensus_vote_metrics"])
    _write_json(group_dir / "agent_weight_metrics.json", summary["agent_weight_metrics"])
    _write_json(group_dir / "consensus_decisions.json", all_decisions)
    _write_json(group_dir / "wbft_results.json", wbft_results)
    return summary


def _run_cases(
    *,
    framework: str,
    runner: Any,
    group: Exp1GroupSpec,
    questions: list[MMLUProQuestion],
    cases_dir: Path,
    traces_dir: Path,
    proposals_dir: Path,
    errors_dir: Path,
    case_retries: int,
    retry_sleep: float,
) -> list[Exp1CaseResult]:
    if framework == "autogen":
        return asyncio.run(
            _run_autogen_cases(
                runner=runner,
                group=group,
                questions=questions,
                cases_dir=cases_dir,
                traces_dir=traces_dir,
                proposals_dir=proposals_dir,
                errors_dir=errors_dir,
                case_retries=case_retries,
                retry_sleep=retry_sleep,
            )
        )
    results: list[Exp1CaseResult] = []
    for index, question in enumerate(questions):
        case_name = f"{index:03d}_{question.question_id}"
        result = _run_case_with_retries(
            runner=runner,
            question=question,
            group=group,
            case_name=case_name,
            retries=case_retries,
            retry_sleep=retry_sleep,
            error_path=errors_dir / f"{case_name}.json",
        )
        _save_case_result(
            result=result,
            case_name=case_name,
            cases_dir=cases_dir,
            traces_dir=traces_dir,
            proposals_dir=proposals_dir,
        )
        results.append(result)
    return results


async def _run_autogen_cases(
    *,
    runner: Exp1Runner,
    group: Exp1GroupSpec,
    questions: list[MMLUProQuestion],
    cases_dir: Path,
    traces_dir: Path,
    proposals_dir: Path,
    errors_dir: Path,
    case_retries: int,
    retry_sleep: float,
) -> list[Exp1CaseResult]:
    results: list[Exp1CaseResult] = []
    for index, question in enumerate(questions):
        case_name = f"{index:03d}_{question.question_id}"
        result = await _run_autogen_case_with_retries(
            runner=runner,
            question=question,
            group=group,
            case_name=case_name,
            retries=case_retries,
            retry_sleep=retry_sleep,
            error_path=errors_dir / f"{case_name}.json",
        )
        _save_case_result(
            result=result,
            case_name=case_name,
            cases_dir=cases_dir,
            traces_dir=traces_dir,
            proposals_dir=proposals_dir,
        )
        results.append(result)
    return results


def _run_case_with_retries(
    *,
    runner: Any,
    question: MMLUProQuestion,
    group: Exp1GroupSpec,
    case_name: str,
    retries: int,
    retry_sleep: float,
    error_path: Path,
) -> Exp1CaseResult:
    attempts = max(0, retries) + 1
    for attempt in range(1, attempts + 1):
        try:
            return runner.run_case(question)
        except Exception as exc:
            _write_json(
                error_path,
                {
                    "case_name": case_name,
                    "group": group.to_dict(),
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if attempt >= attempts:
                raise
            print(
                f"[{group.group_id}] {case_name} failed on attempt "
                f"{attempt}/{attempts}: {type(exc).__name__}: {exc}. Retrying..."
            )
            time.sleep(retry_sleep)
    raise RuntimeError(f"Failed to run case after {attempts} attempts: {case_name}")


async def _run_autogen_case_with_retries(
    *,
    runner: Exp1Runner,
    question: MMLUProQuestion,
    group: Exp1GroupSpec,
    case_name: str,
    retries: int,
    retry_sleep: float,
    error_path: Path,
) -> Exp1CaseResult:
    attempts = max(0, retries) + 1
    for attempt in range(1, attempts + 1):
        try:
            return await runner.run_case_async(question)
        except Exception as exc:
            _write_json(
                error_path,
                {
                    "case_name": case_name,
                    "group": group.to_dict(),
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if attempt >= attempts:
                raise
            print(
                f"[{group.group_id}] {case_name} failed on attempt "
                f"{attempt}/{attempts}: {type(exc).__name__}: {exc}. Retrying..."
            )
            await asyncio.sleep(retry_sleep)
    raise RuntimeError(f"Failed to run case after {attempts} attempts: {case_name}")


def _save_case_result(
    *,
    result: Exp1CaseResult,
    case_name: str,
    cases_dir: Path,
    traces_dir: Path,
    proposals_dir: Path,
) -> None:
    _write_json(cases_dir / f"{case_name}.json", result.to_case_json())
    (traces_dir / f"{case_name}.txt").write_text(result.trace, encoding="utf-8")
    _write_json(proposals_dir / f"{case_name}.json", result.memory_proposals)


def _memory_proposal_metrics(results: list[Exp1CaseResult]) -> dict[str, Any]:
    proposals = [proposal for result in results for proposal in result.memory_proposals]
    lifecycles = [
        proposal.get("_experiment", {}).get("lifecycle", "unknown")
        for proposal in proposals
    ]
    token_counts = [
        int(proposal.get("_experiment", {}).get("token_count", 0) or 0)
        for proposal in proposals
    ]
    return {
        "proposal_count": len(proposals),
        "accepted_count": lifecycles.count("consensus_accepted"),
        "rejected_count": lifecycles.count("consensus_rejected"),
        "self_rejected_count": lifecycles.count("self_rejected"),
        "lifecycle_counts": {
            lifecycle: lifecycles.count(lifecycle)
            for lifecycle in sorted(set(lifecycles))
        },
        "memory_proposal_tokens": sum(token_counts),
        "per_proposal_tokens": token_counts,
        "per_case_counts": {
            result.case_id: len(result.memory_proposals)
            for result in results
        },
    }


def _consensus_vote_metrics(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    qc_values = []
    voter_counts = []
    for decision in decisions:
        metadata = dict(decision.get("metadata", {}) or {})
        quorum = metadata.get("quorum") or metadata.get("quorum_certificate") or {}
        qc = metadata.get("qc", quorum.get("qc") if isinstance(quorum, dict) else None)
        if qc is not None:
            qc_values.append(float(qc))
        voter_counts.append(len(decision.get("votes", []) or []))
    return {
        "decision_count": len(decisions),
        "qc_values": qc_values,
        "qc_min": min(qc_values) if qc_values else None,
        "qc_mean": sum(qc_values) / len(qc_values) if qc_values else None,
        "qc_max": max(qc_values) if qc_values else None,
        "voter_counts": voter_counts,
        "voter_count_min": min(voter_counts) if voter_counts else None,
        "voter_count_mean": sum(voter_counts) / len(voter_counts) if voter_counts else None,
        "voter_count_max": max(voter_counts) if voter_counts else None,
    }


def _agent_weight_metrics(decisions: list[dict[str, Any]], group_dir: Path) -> dict[str, Any]:
    snapshots = []
    for decision in decisions:
        metadata = dict(decision.get("metadata", {}) or {})
        snapshot = metadata.get("weight_snapshots_after_update")
        if snapshot:
            snapshots.append(
                {
                    "proposal_id": decision.get("proposal_id"),
                    "weight_snapshots_after_update": snapshot,
                }
            )
    process_snapshots = _weight_snapshots_from_process_log(group_dir)
    return {
        "update_count": len(snapshots),
        "weight_snapshots_after_update": snapshots,
        "process_log_weight_snapshots": process_snapshots,
    }


def _weight_snapshots_from_process_log(group_dir: Path) -> list[dict[str, Any]]:
    snapshots = []
    for name in ("consensus_process.jsonl", "crewai_consensus_process.jsonl"):
        path = group_dir / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            for key in ("initial_weight_snapshots", "weight_snapshots", "final_weight_snapshots"):
                if key in payload:
                    snapshots.append(
                        {
                            "event": payload.get("event"),
                            "case_id": payload.get("case_id"),
                            "snapshot_type": key,
                            "weights": payload[key],
                        }
                    )
    return snapshots


def _wbft_metrics(wbft_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "result_count": len(wbft_results),
        "participant_counts": [
            result.get("participant_count")
            for result in wbft_results
            if result.get("participant_count") is not None
        ],
    }


def _group_dir_name(framework: str, group: Exp1GroupSpec) -> str:
    return f"{framework}__{group.group_id}"


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "framework",
        "method",
        "n",
        "f",
        "model_regime",
        "cases",
        "correct",
        "accuracy",
        "total_seconds",
        "task_completion_seconds",
        "consensus_extra_seconds",
        "interaction_message_count",
        "extra_consensus_or_wbft_message_count",
        "proposal_count",
        "accepted_proposal_count",
        "rejected_proposal_count",
        "decision_count",
        "qc_mean",
        "voter_count_mean",
        "total_tokens",
        "memory_proposal_tokens",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            group = summary["group"]
            regular = summary.get("regular_metrics", {})
            time_metrics = regular.get("time", {})
            interaction = regular.get("interaction", {})
            tokens = regular.get("tokens", {})
            proposal_metrics = summary.get("memory_proposal_metrics", {})
            vote_metrics = summary.get("consensus_vote_metrics", {})
            writer.writerow(
                {
                    "framework": summary.get("framework"),
                    "method": group.get("method"),
                    "n": group.get("n"),
                    "f": group.get("f"),
                    "model_regime": group.get("model_regime"),
                    "cases": summary.get("cases"),
                    "correct": summary.get("correct"),
                    "accuracy": summary.get("accuracy"),
                    "total_seconds": time_metrics.get("total_seconds"),
                    "task_completion_seconds": time_metrics.get("task_completion_seconds"),
                    "consensus_extra_seconds": time_metrics.get("consensus_extra_seconds"),
                    "interaction_message_count": interaction.get("agent_message_count"),
                    "extra_consensus_or_wbft_message_count": interaction.get(
                        "extra_consensus_or_wbft_message_count"
                    ),
                    "proposal_count": proposal_metrics.get("proposal_count"),
                    "accepted_proposal_count": proposal_metrics.get("accepted_count"),
                    "rejected_proposal_count": proposal_metrics.get("rejected_count"),
                    "decision_count": vote_metrics.get("decision_count"),
                    "qc_mean": vote_metrics.get("qc_mean"),
                    "voter_count_mean": vote_metrics.get("voter_count_mean"),
                    "total_tokens": tokens.get("total_tokens"),
                    "memory_proposal_tokens": proposal_metrics.get("memory_proposal_tokens"),
                }
            )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
