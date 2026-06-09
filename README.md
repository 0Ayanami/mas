# Consensus Memory MAS

This repository contains a first-pass multi-agent framework for the research direction in:

`theory_base/0-reseach proposal：基于共识机制的多智能体记忆抗拜占庭同步.md`

The framework is intentionally small but extensible:

- CAMEL `ChatAgent` adapter for LLM-backed agents.
- Tool registry with file-reading and memory-search examples.
- Structured `MemoryProposal` and `VerificationVector` models.
- A basic smart-quorum consensus decision module.
- A runnable research-loop demo.

# Development Log
| date | dev | relative-docs|
| ---  | --- | --- |
| 2026-06-04 | 初始版本 | `README.md` |
| 2026-06-05 | 使用camel-ai 搭建multi-agent原型框架 | `src/mas_framework/tools.py`,`agents.py`,`orchestrator.py` |
| 2026-06-06 | memory模块完善 | `src/mas_framework/memory.py` |
| 2026-06-07 | 基础数据类 | `src/mas_framework/models.py` |
| 2026-06-08 | 共识机制完善 | `src/mas_framework/consensus.py` |
| 2026-06-09 | 验证过程梳理 | `src/mas_framework/orchestrator.py` |

## Quick Start

```powershell
uv venv
uv pip install -e ".[dev]"
copy .env.example .env

pytest
```

To enable CAMEL-backed agents:

```powershell
uv pip install -e ".[camel,dev]"
```

## Project Shape

- `src/mas_framework/models.py`: shared schemas for proposals, verification, agents, and decisions.
- `src/mas_framework/agents.py`: CAMEL adapter.
- `src/mas_framework/tools.py`: tool registry and default research tools.
- `src/mas_framework/memory.py`: persistent shared memory store.
- `src/mas_framework/consensus.py`: current quorum and decision mechanism.
- `src/mas_framework/orchestrator.py`: multi-agent workflow.
- `examples/run_research_demo.py`: executable demo.

