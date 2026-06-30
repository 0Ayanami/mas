from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

from mas_framework.orchestrator import Orchestrator


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the AutoGen MAS research workflow.")
    parser.add_argument(
        "--document",
        default="theory_base/0-reseach proposal：基于共识机制的多智能体记忆抗拜占庭同步(1).md",
        help="Path to the research markdown file.",
    )
    parser.add_argument(
        "--agent-id",
        default="researcher_1",
        help="Agent that performs the research step and proposes memory.",
    )
    parser.add_argument(
        "--task-id",
        default="consensus-memory-research",
        help="Task ID stored in the memory proposal.",
    )
    args = parser.parse_args()

    document_path = Path(args.document)
    if not document_path.exists():
        raise FileNotFoundError(document_path)
    document = document_path.read_text(encoding="utf-8")

    orchestrator = Orchestrator()
    agent = orchestrator.agents.get(args.agent_id)
    if agent is None:
        raise RuntimeError(f"Agent is unavailable: {args.agent_id}")

    task_prompt = (
        "Analyze the following research plan and identify the most useful next implementation "
        "step for the consensus-memory MAS.\n\n"
        f"{document[:20_000]}"
    )
    react_trace = agent.run(task_prompt)
    proposal = orchestrator.propose_memory_from_trace(
        agent_id=args.agent_id,
        task_id=args.task_id,
        react_trace=react_trace,
        task_context=f"Research document: {document_path}",
    )

    print(json.dumps(
        {
            "react_trace": react_trace,
            "proposal": proposal.model_dump() if proposal else None,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
