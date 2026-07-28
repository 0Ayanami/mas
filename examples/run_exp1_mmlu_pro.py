from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mas_framework.exp_1.exp1_mmlu import (
    Exp1CaseResult,
    Exp1Config,
    Exp1GroupSpec,
    Exp1Runner,
    MMLUProQuestion,
    load_mmlu_pro_questions,
    summarize_group_results,
)
from mas_framework.memory import Mem0MemoryBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp1 MMLU-Pro fault-tolerance experiments.")
    parser.add_argument(
        "--config",
        default="src/mas_framework/configs/experiment_configs/exp1_mmlu_pro.yaml",
        help="Exp1 config YAML.",
    )
    parser.add_argument("--dataset-path", default=None, help="Override MMLU-Pro local path.")
    parser.add_argument("--method", default=None, help="Run only one method.")
    parser.add_argument("--methods", default=None, help="Comma-separated methods to run.")
    parser.add_argument("--n", type=int, default=None, help="Run only one agent count.")
    parser.add_argument("--f", type=int, default=None, help="Run only one Byzantine count.")
    parser.add_argument("--f-values", default=None, help="Comma-separated Byzantine counts to run.")
    parser.add_argument("--model-regime", default=None, help="Run only one model regime.")
    parser.add_argument("--limit-cases", type=int, default=None, help="Limit sampled cases.")
    parser.add_argument("--limit-groups", type=int, default=None, help="Limit experiment groups.")
    parser.add_argument("--run-id", default=None, help="Override run id.")
    parser.add_argument(
        "--memory-user-id-prefix",
        default=None,
        help="Prefix for group-level Mem0 user ids. Defaults to run id.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip groups that already have summary.json.")
    parser.add_argument("--case-retries", type=int, default=2, help="Retries per failed case.")
    parser.add_argument("--retry-sleep", type=float, default=10.0, help="Seconds to sleep between retries.")
    parser.add_argument("--dry-run", action="store_true", help="Plan groups and sample dataset without API calls.")
    args = parser.parse_args()

    config = Exp1Config.from_yaml(args.config)
    if args.dataset_path:
        config = _replace_config(config, dataset_path=args.dataset_path)
    run_id = args.run_id or _run_id(config.run_id)
    run_dir = Path(config.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config_snapshot.yaml")

    questions = load_mmlu_pro_questions(
        config.dataset_path,
        category_limit=config.category_count,
        samples_per_category=config.samples_per_category,
        seed=config.seed,
    )
    if args.limit_cases is not None:
        questions = questions[: args.limit_cases]
    _write_json(run_dir / "dataset_sample.json", [question.to_dict() for question in questions])

    groups = _filter_groups(
        config.group_specs,
        method=args.method,
        methods=_csv_values(args.methods),
        n=args.n,
        f=args.f,
        f_values=_csv_ints(args.f_values),
        model_regime=args.model_regime,
    )
    if args.limit_groups is not None:
        groups = groups[: args.limit_groups]

    if args.dry_run:
        payload = {
            "run_dir": str(run_dir),
            "cases": len(questions),
            "groups": [group.to_dict() for group in groups],
        }
        _write_json(run_dir / "dry_run_plan.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    all_summaries: list[dict[str, Any]] = []
    for group in groups:
        group_dir = run_dir / "groups" / group.group_id
        summary_path = group_dir / "summary.json"
        if args.resume and summary_path.exists():
            all_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        summary = _run_group(
            config=config,
            group=group,
            questions=questions,
            group_dir=group_dir,
            memory_user_id_prefix=args.memory_user_id_prefix or run_id,
            resume=args.resume,
            case_retries=args.case_retries,
            retry_sleep=args.retry_sleep,
        )
        all_summaries.append(summary)
        _write_json(run_dir / "all_results.json", all_summaries)
        _write_csv(run_dir / "all_results.csv", all_summaries)
        print(
            f"[{group.group_id}] accuracy={summary['accuracy']:.4f} "
            f"cases={summary['cases']} time={summary['time']['total_seconds']:.2f}s"
        )

    _write_json(run_dir / "all_results.json", all_summaries)
    _write_csv(run_dir / "all_results.csv", all_summaries)
    print(f"Results written to: {run_dir}")


def _run_group(
    *,
    config: Exp1Config,
    group: Exp1GroupSpec,
    questions: list[Any],
    group_dir: Path,
    memory_user_id_prefix: str,
    resume: bool,
    case_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    cases_dir = group_dir / "cases"
    traces_dir = group_dir / "traces"
    proposals_dir = group_dir / "memory_proposals"
    errors_dir = group_dir / "errors"
    for directory in (cases_dir, traces_dir, proposals_dir, errors_dir):
        directory.mkdir(parents=True, exist_ok=True)
    memory_user_id = f"{memory_user_id_prefix}__{group.group_id}"
    memory_backend = (
        Mem0MemoryBackend(
            topk=config.memory_topk,
            default_user_id=memory_user_id,
        )
        if group.uses_ours
        else None
    )
    _write_json(
        group_dir / "memory_user_id.json",
        {
            "memory_user_id": memory_user_id if group.uses_ours else None,
            "uses_mem0": group.uses_ours,
        },
    )
    runner = Exp1Runner(
        config=config,
        memory_backend=memory_backend,
        memory_user_id=memory_user_id if group.uses_ours else None,
    )
    results = []
    all_decisions: list[dict[str, Any]] = []
    wbft_results: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        case_name = f"{index:03d}_{question.question_id}"
        case_path = cases_dir / f"{case_name}.json"
        trace_path = traces_dir / f"{case_name}.txt"
        proposals_path = proposals_dir / f"{case_name}.json"
        error_path = errors_dir / f"{case_name}.json"
        if resume and case_path.exists():
            result = _load_completed_case(
                question=question,
                case_path=case_path,
                trace_path=trace_path,
                proposals_path=proposals_path,
            )
        else:
            result = _run_case_with_retries(
                runner=runner,
                question=question,
                group=group,
                case_name=case_name,
                retries=case_retries,
                retry_sleep=retry_sleep,
                error_path=error_path,
            )
            _write_json(case_path, result.to_case_json())
            trace_path.write_text(result.trace, encoding="utf-8")
            _write_json(proposals_path, result.memory_proposals)
            if error_path.exists():
                error_path.unlink()
        results.append(result)
        all_decisions.extend(result.consensus_decisions)
        if result.wbft_result is not None:
            wbft_results.append({"case_id": result.case_id, **result.wbft_result})
    summary = summarize_group_results(group, results)
    _write_json(group_dir / "summary.json", summary)
    _write_json(group_dir / "regular_metrics.json", summary["regular_metrics"])
    _write_json(
        group_dir / "memory_proposal_metrics.json",
        summary["memory_proposal_metrics"],
    )
    _write_json(group_dir / "consensus_decisions.json", all_decisions)
    _write_json(group_dir / "wbft_results.json", wbft_results)
    return summary


def _run_case_with_retries(
    *,
    runner: Exp1Runner,
    question: Any,
    group: Exp1GroupSpec,
    case_name: str,
    retries: int,
    retry_sleep: float,
    error_path: Path,
) -> Exp1CaseResult:
    attempts = max(0, retries) + 1
    for attempt in range(1, attempts + 1):
        try:
            return runner.run_case(question, group)
        except Exception as exc:
            _write_json(
                error_path,
                {
                    "case_name": case_name,
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


def _load_completed_case(
    *,
    question: Any,
    case_path: Path,
    trace_path: Path,
    proposals_path: Path,
) -> Exp1CaseResult:
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    trace = trace_path.read_text(encoding="utf-8") if trace_path.exists() else ""
    proposals = (
        json.loads(proposals_path.read_text(encoding="utf-8"))
        if proposals_path.exists()
        else []
    )
    return Exp1CaseResult(
        case_id=str(payload.get("case_id", question.question_id)),
        question=question if isinstance(question, MMLUProQuestion) else question,
        predicted_answer=payload.get("predicted_answer"),
        correct=bool(payload.get("correct", False)),
        final_text=str(payload.get("final_text", "")),
        trace=trace,
        metrics=dict(payload.get("metrics", {})),
        agent_specs=dict(payload.get("agent_specs", {})),
        consensus_decisions=[],
        memory_proposals=proposals if isinstance(proposals, list) else [],
        memory_uploads=list(payload.get("memory_uploads", [])),
        wbft_result=payload.get("wbft_result"),
    )


def _filter_groups(
    groups: list[Exp1GroupSpec],
    *,
    method: str | None,
    methods: set[str] | None,
    n: int | None,
    f: int | None,
    f_values: set[int] | None,
    model_regime: str | None,
) -> list[Exp1GroupSpec]:
    selected = []
    for group in groups:
        if method is not None and group.method != method:
            continue
        if methods is not None and group.method not in methods:
            continue
        if n is not None and group.n != n:
            continue
        if f is not None and group.f != f:
            continue
        if f_values is not None and group.f not in f_values:
            continue
        if model_regime is not None and group.model_regime != model_regime:
            continue
        selected.append(group)
    return selected


def _csv_values(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _csv_ints(value: str | None) -> set[int] | None:
    values = _csv_values(value)
    if values is None:
        return None
    return {int(item) for item in values}


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
        "consensus_extra_message_count",
        "proposal_count",
        "decision_count",
        "qc_mean",
        "voter_count_mean",
        "total_tokens",
        "memory_proposal_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            group = summary["group"]
            writer.writerow(
                {
                    "method": group["method"],
                    "n": group["n"],
                    "f": group["f"],
                    "model_regime": group["model_regime"],
                    "cases": summary["cases"],
                    "correct": summary["correct"],
                    "accuracy": summary["accuracy"],
                    "total_seconds": summary["time"]["total_seconds"],
                    "task_completion_seconds": summary["time"]["task_completion_seconds"],
                    "consensus_extra_seconds": summary["time"]["consensus_extra_seconds"],
                    "interaction_message_count": summary["messages"]["interaction_message_count"],
                    "consensus_extra_message_count": summary["messages"]["consensus_extra_message_count"],
                    "proposal_count": summary["memory"]["proposal_count"],
                    "decision_count": summary["consensus"]["decision_count"],
                    "qc_mean": summary["consensus"]["qc_mean"],
                    "voter_count_mean": summary["consensus"]["voter_count_mean"],
                    "total_tokens": summary["tokens"]["total_tokens"],
                    "memory_proposal_tokens": summary["tokens"]["memory_proposal_tokens"],
                }
            )


def _replace_config(config: Exp1Config, **updates: Any) -> Exp1Config:
    from dataclasses import replace

    return replace(config, **updates)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
