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

try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    Console = None
    Panel = None

from mas_framework.orchestrator import ResearchOrchestrator
from mas_framework.memory import SQLiteMemoryStore


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the initial CAMEL MAS research workflow.")
    parser.add_argument(
        "--document",
        default="D:/markdowns/agent/security/基于共识机制的多智能体记忆抗拜占庭同步.md",
        help="Path to the research markdown file.",
    )
    parser.add_argument("--db", default="data/memory.sqlite", help="SQLite memory path.")
    args = parser.parse_args()

    document_path = Path(args.document)
    if not document_path.exists():
        fallback = Path(__file__).resolve().parents[1] / "docs" / "research_brief.md"
        print(f"Document not found: {document_path}. Using local fallback: {fallback}")
        document_path = fallback

    console = Console() if Console else None
    orchestrator = ResearchOrchestrator(memory=SQLiteMemoryStore(db_path=args.db))
    proposal, decision = orchestrator.run_document_research(document_path=str(document_path))

    if console and Panel:
        console.print(Panel.fit(proposal.short_label(), title="Proposal"))
        console.print_json(proposal.model_dump_json())
        console.print(Panel.fit(decision.rationale, title=f"Decision: {decision.status.value}"))
    else:
        print(f"Proposal: {proposal.short_label()}")
        print(proposal.model_dump_json(indent=2))
        print(f"Decision: {decision.status.value} - {decision.rationale}")

    export_path = Path(args.db).with_name("latest_decision.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        json.dumps(
            {
                "proposal": proposal.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if console:
        console.print(f"Saved latest decision to {export_path}")
    else:
        print(f"Saved latest decision to {export_path}")


if __name__ == "__main__":
    main()
