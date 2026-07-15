"""Command-line entry point for WBFT AutoGen experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from mas_framework.wbft import WBFTRunConfig, WBFTRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the WBFT agent-output consensus baseline on TAMAS-style AutoGen cases."
    )
    parser.add_argument("--json-data", required=True, help="Path to a TAMAS JSON dataset file.")
    parser.add_argument(
        "--config",
        default="src/mas_framework/wbft/configs/wbft_config.yaml",
        help="WBFT-specific experiment config YAML.",
    )
    parser.add_argument(
        "--mode",
        choices=["round_robin", "magentic_one"],
        default=None,
        help="AutoGen team workflow.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Number of cases to run.")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset case index to start from.")
    parser.add_argument("--max-messages", type=int, default=None, help="Override AutoGen max messages.")
    parser.add_argument("--model", default=None, help="Override all agents to use this model.")
    parser.add_argument("--honest-model", default=None, help="Override honest agent model.")
    parser.add_argument("--byzantine-model", default=None, help="Override Byzantine agent model.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Ignore low-confidence votes per answer when possible.",
    )
    parser.add_argument(
        "--normalization",
        choices=["auto", "number", "none"],
        default=None,
        help="Answer normalization mode before voting.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Optional directory for per-case .txt traces compatible with TAMAS eval scripts.",
    )
    args = parser.parse_args()

    config = WBFTRunConfig.from_config(args.config, mode=args.mode)
    updates: dict[str, Any] = {}
    if args.model:
        updates["model"] = args.model
        updates["honest_model"] = args.model
        updates["byzantine_model"] = args.model
    if args.honest_model:
        updates["honest_model"] = args.honest_model
    if args.byzantine_model:
        updates["byzantine_model"] = args.byzantine_model
    if args.confidence_threshold is not None:
        updates["confidence_threshold"] = args.confidence_threshold
    if args.normalization is not None:
        updates["normalization"] = args.normalization
    if args.max_messages is not None:
        updates["max_messages"] = args.max_messages
    if updates:
        config = replace(config, **updates)

    runner = WBFTRunner(config=config)
    cases = runner.load_dataset(args.json_data)
    end_index = min(len(cases), args.start_index + args.limit)
    selected_cases = list(enumerate(cases[args.start_index:end_index], start=args.start_index))
    results = []

    output = Path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)
    domain = _domain_name(Path(args.json_data))

    for case_index, case in selected_cases:
        result = runner.run_case(case, task_id=f"{domain}_byzantine-{case_index}")
        results.append(result)
        if trace_dir:
            (trace_dir / f"{domain}_{case_index}.txt").write_text(
                result.trace,
                encoding="utf-8",
            )
        payload = [item.to_dict() for item in results]
        if output:
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{domain}_{case_index}] wbft_responses={len(result.agent_responses)}")

    payload = [result.to_dict() for result in results]
    if not output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

def _domain_name(dataset_file: Path) -> str:
    name = dataset_file.stem
    for suffix in ("_byzantine", "_colluding", "_contradicting"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


if __name__ == "__main__":
    main()
