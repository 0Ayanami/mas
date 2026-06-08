# Consensus Memory MAS

This repository contains a first-pass multi-agent framework for the research direction in:

`theory_base/0-reseach proposal：基于共识机制的多智能体记忆抗拜占庭同步.md`

The framework is intentionally small but extensible:

- CAMEL `ChatAgent` adapter for LLM-backed agents.
- Deterministic fallback agents so the framework can run without API keys.
- Tool registry with file-reading and memory-search examples.
- Structured `MemoryProposal` and `VerificationVector` models.
- A basic smart-quorum consensus decision module.
- A runnable research-loop demo.

## Quick Start

```powershell
uv venv
uv pip install -e ".[dev]"
copy .env.example .env
python examples/run_research_demo.py --document "D:/markdowns/agent/security/基于共识机制的多智能体记忆抗拜占庭同步.md"
pytest
```

If model credentials are configured, agents will use CAMEL. Without credentials, the demo uses a deterministic local fallback so the architecture can still be inspected and tested.
If the external markdown path is not available, the demo falls back to `docs/research_brief.md`.

To enable CAMEL-backed agents:

```powershell
uv pip install -e ".[camel,dev]"
```

## Project Shape

- `src/mas_framework/models.py`: shared schemas for proposals, verification, agents, and decisions.
- `src/mas_framework/agents.py`: CAMEL adapter and fallback agent implementation.
- `src/mas_framework/tools.py`: tool registry and default research tools.
- `src/mas_framework/memory.py`: persistent shared memory store.
- `src/mas_framework/consensus.py`: current quorum and decision mechanism.
- `src/mas_framework/orchestrator.py`: multi-agent research workflow.
- `examples/run_research_demo.py`: executable demo.

## Next Research Steps

1. Replace the current threshold policy with protocol-specific strategies: PBFT, HotStuff, Raft-like leader flow, and gossip dissemination.
2. Expand `VerificationVector` into a calibrated verifier with retrieval-backed fact checks and injection/poisoning classifiers.
3. Add adversarial agent profiles for Byzantine behavior simulation.
4. Add evaluation harnesses for robustness, task quality, and communication efficiency.
