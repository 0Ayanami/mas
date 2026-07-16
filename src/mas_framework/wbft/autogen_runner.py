"""AutoGen workflow for WBFT agent-output consensus."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm

from mas_framework.common import (
    _agent_outputs,
    _final_text,
    _normalize_agent_name,
    build_model_client,
    run_autogen_sync,
)
from mas_framework.metrics import message_count, token_usage
from mas_framework.tamas_data import load_tamas_dataset
from mas_framework.tamas_workflow import TAMASToolLoader
from mas_framework.wbft.consensus import WBFTConsensus, parse_wbft_response
from mas_framework.wbft.models import (
    WBFTAgentResponse,
    WBFTAgentSpec,
    WBFTConsensusResult,
    WBFTRunConfig,
)


@dataclass
class WBFTRunResult:
    final_text: str
    events: list[Any]
    agent_responses: list[WBFTAgentResponse] = field(default_factory=list)
    consensus_result: WBFTConsensusResult | None = None
    run_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def trace(self) -> str:
        parts: list[str] = []
        for event in self.events:
            source = getattr(event, "source", None)
            content = getattr(event, "content", None)
            if source and isinstance(content, str):
                parts.append(f"{source}: {content}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_text": self.final_text,
            "agent_responses": [response.to_dict() for response in self.agent_responses],
            "consensus_result": (
                self.consensus_result.to_dict()
                if self.consensus_result is not None
                else None
            ),
            "run_metrics": dict(self.run_metrics),
        }


class WBFTRunner:
    """Run TAMAS-style AutoGen teams, then apply WBFT over agent outputs."""

    def __init__(
        self,
        *,
        config: WBFTRunConfig | None = None,
        tamas_root: str | Path = "TAMAS-main",
        model_client: Any | None = None,
    ) -> None:
        self.config = config or WBFTRunConfig()
        self.tool_loader = TAMASToolLoader(tamas_root)
        self.model_client = model_client
        self._model_clients: dict[str, Any] = {}
        self._agent_specs: dict[str, WBFTAgentSpec] = {}

    def _client_for_model(self, model: str | None) -> Any:
        if self.model_client is not None:
            return self.model_client
        cache_key = model or "__env_model__"
        if cache_key not in self._model_clients:
            self._model_clients[cache_key] = build_model_client(
                model=model,
            )
        return self._model_clients[cache_key]

    @staticmethod
    def load_dataset(path: str | Path) -> list[dict[str, Any]]:
        return load_tamas_dataset(path)

    def run_dataset(
        self,
        path: str | Path,
        *,
        limit: int | None = None,
    ) -> list[WBFTRunResult]:
        cases = self.load_dataset(path)
        if limit is not None:
            cases = cases[:limit]
        return [
            self.run_case(case, task_id=f"{Path(path).stem}-{index}")
            for index, case in enumerate(cases)
        ]

    def run_case(
        self,
        case: dict[str, Any],
        *,
        task_id: str = "wbft-task",
    ) -> WBFTRunResult:
        return run_autogen_sync(self.run_case_async(case, task_id=task_id))

    async def run_case_async(
        self,
        case: dict[str, Any],
        *,
        task_id: str,
    ) -> WBFTRunResult:
        agents = self._build_agents(case)
        team = self._build_team(agents)
        events: list[Any] = []
        case_started = time.perf_counter()

        async for event in team.run_stream(task="Task: " + case["user query"]):
            events.append(event)

        agent_responses = self._collect_agent_responses(events)
        consensus = WBFTConsensus(
            confidence_threshold=self.config.confidence_threshold,
            convergence_threshold=self.config.convergence_threshold,
            fault_tolerance_threshold=self.config.fault_tolerance_threshold,
            minimum_participants=self.config.minimum_participants,
            normalization=self.config.normalization,
        )
        consensus_started = time.perf_counter()
        consensus_result = consensus.decide(agent_responses)
        consensus_elapsed = time.perf_counter() - consensus_started

        return WBFTRunResult(
            final_text=_final_text(events),
            events=events,
            agent_responses=agent_responses,
            consensus_result=consensus_result,
            run_metrics={
                "total_case_seconds": time.perf_counter() - case_started,
                "consensus_extra_seconds": consensus_elapsed,
                "interaction_message_count": message_count(events),
                "consensus_extra_message_count": 0,
                "task_id": task_id,
                **token_usage(events=events, trace=_trace(events), proposals=[]),
            },
        )

    def _build_agents(self, case: dict[str, Any]) -> list[AssistantAgent]:
        agent_list: list[AssistantAgent] = []
        self._agent_specs = {}
        agent_ids = [
            _normalize_agent_name(f"{item['agent_name']}_{index}")
            for index, item in enumerate(case["agents"], start=1)
        ]
        for index, item in enumerate(case["agents"], start=1):
            display_name = item["agent_name"]
            agent_id = agent_ids[index - 1]
            is_byzantine = self._is_byzantine_agent(item)
            model = self._model_for_agent(is_byzantine)
            self._agent_specs[agent_id] = WBFTAgentSpec(
                agent_id=agent_id,
                display_name=display_name,
                model=model,
                is_byzantine=is_byzantine,
            )

            system_message = item["agent_description"] + self._wbft_response_instruction()
            agent_list.append(
                AssistantAgent(
                    name=agent_id,
                    description=display_name,
                    model_client=self._client_for_model(model),
                    tools=self.tool_loader.tools_for_agent(display_name),
                    handoffs=[target for target in agent_ids if target != agent_id],
                    system_message=system_message,
                )
            )
        return agent_list

    def _build_team(self, agents: list[AssistantAgent]) -> Any:
        termination = MaxMessageTermination(self.config.max_messages) | TextMentionTermination(
            "TERMINATE"
        )
        return Swarm(
            agents,
            termination_condition=termination,
            max_turns=self.config.max_messages,
        )

    def _collect_agent_responses(
        self,
        events: list[Any],
    ) -> list[WBFTAgentResponse]:
        latest_by_agent: dict[str, WBFTAgentResponse] = {}
        for source, content in _agent_outputs(events):
            if source not in self._agent_specs:
                continue
            response = parse_wbft_response(
                source,
                content,
                confidence_extraction_method=self.config.confidence_extraction_method,
                include_unstructured=self.config.include_unstructured_outputs,
                fallback_confidence=self.config.fallback_confidence,
            )
            if response is not None:
                latest_by_agent[source] = response
        return [
            latest_by_agent[agent_id]
            for agent_id in self._agent_specs
            if agent_id in latest_by_agent
        ]

    def _is_byzantine_agent(self, item: dict[str, Any]) -> bool:
        agent_type = str(item.get("agent_type", "")).strip().lower()
        if agent_type == "byzantine":
            return True
        if agent_type == "honest":
            return False
        raise ValueError(
            "WBFT agents must include agent_type='honest' or agent_type='byzantine'. "
            f"Agent {item.get('agent_name', '<unknown>')!r} has agent_type={item.get('agent_type')!r}."
        )

    def _model_for_agent(self, is_byzantine: bool) -> str | None:
        if is_byzantine:
            return self.config.byzantine_model or self.config.model
        return self.config.honest_model or self.config.model

    def _wbft_response_instruction(self) -> str:
        return (
            "\n\nAt the end of your final relevant task response, include exactly one "
            "prompt-level WBFT confidence report. This is for external Byzantine fault "
            "tolerant voting over agent outputs; do not mention shared memory.\n\n"
            "Use this format exactly:\n"
            "Answer: [your concise final answer or decision]\n"
            "Confidence: [a number from 0.00 to 1.00]\n"
            "Reasoning: [brief justification]\n\n"
            f"{self.config.confidence_instruction}\n\n"
            "Also include this machine-readable copy when possible:\n"
            "WBFT_RESPONSE\n"
            "```json\n"
            "{\n"
            '  "answer": "your concise final answer or decision",\n'
            '  "confidence": 0.0,\n'
            '  "reasoning": "brief justification"\n'
            "}\n"
            "```\n"
            "END_WBFT_RESPONSE\n"
            "The confidence must be a number from 0.0 to 1.0."
        )


def _trace(events: list[Any]) -> str:
    parts: list[str] = []
    for event in events:
        source = getattr(event, "source", None)
        content = getattr(event, "content", None)
        if source and isinstance(content, str):
            parts.append(f"{source}: {content}")
    return "\n\n".join(parts)
