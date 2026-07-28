from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mas_framework.exp3.tamas_workflow import TAMASAutoGenRunner, TAMASRunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TAMAS with AutoGen, Mem0 memory, and optional consensus.")
    parser.add_argument("--json-data", required=True, help="Path to a TAMAS JSON dataset file.")
    parser.add_argument(
        "--config",
        default="src/mas_framework/configs/experiment_configs/unified_config.yaml",
        help="Unified experiment config YAML.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Number of TAMAS cases to run.")
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument("--consensus", action="store_true", help="Enable memory proposal consensus.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    config = TAMASRunConfig.from_unified_config(args.config)
    if args.model:
        config = _replace_config(config, model=args.model)
    if args.consensus:
        config = _replace_config(config, consensus_enabled=True)

    runner = TAMASAutoGenRunner(config=config)
    results = runner.run_dataset(args.json_data, limit=args.limit)
    payload = [
        {
            "final_text": item.final_text,
            "trace": item.trace,
            "consensus_decisions": [
                decision.to_dict() if hasattr(decision, "to_dict") else str(decision)
                for decision in item.consensus_decisions
            ],
        }
        for item in results
    ]

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def _replace_config(config: TAMASRunConfig, **updates):
    values = config.__dict__.copy()
    values.update(updates)
    return TAMASRunConfig(**values)


if __name__ == "__main__":
    main()
