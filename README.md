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

#TODO 系统的工作流
整个系统模拟分布式的多智能体协作系统，每个agent都有自己的任务和状态，通过共识机制协调和决策。

#TODO 什么时候提案memory？
通过prompt让agent决定什么时候提案memory。agent在完成一步任务后，显式的返回一个flag，工作流中捕获这一flag，此时进入memory proposal的构建阶段。

#TODO proposal的构建过程
首先基于规则的方法构建proposal的header，然后使用prompt让agent自行总结重要信息，即proposal的body。再对proposal进行自我验证和打分，confidence超过一个阈值后，proposal被提交给其他agent进行验证共识过程。


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

