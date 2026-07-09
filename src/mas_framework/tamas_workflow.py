from __future__ import annotations

import asyncio
import importlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import MagenticOneGroupChat, RoundRobinGroupChat

from mas_framework.common import (
    _normalize_agent_name,
    _base_agent_name,
    _agent_outputs,
    _final_text,
    build_model_client,
    run_autogen_sync,
)
from mas_framework.consensus import (
    AutoGenProposalEvaluator,
    HeuristicProposalEvaluator,
    MajorityVoteConsensus,
    ProposalBuilder,
    SmartQuorumConsensus,
    VerificationContext,
    VerificationEngine,
    WeightManager,
)
from mas_framework.memory import Mem0MemoryBackend, build_memory_tools


TAMASMode = Literal["round_robin", "magentic_one"]


AGENT_TOOL_MODULES = {
    "Adaptive Learning Agent": "tools_adaptive_learning_agent",
    "Assessment Agent": "tools_assessment_agent",
    "Progress Tracking Agent": "tools_progress_tracking_agent",
    "Concept Clarification Agent": "tools_concept_clarification_agent",
    "Identity Verification Agent": "tools_identity_verification_agent",
    "Market Analysis Agent": "tools_market_analysis_agent",
    "Trade Execution Agent": "tools_trade_execution_agent",
    "Risk Management Agent": "tools_risk_management_agent",
    "Diagnosis Agent": "tools_diagnosis_agent",
    "Medication Adherence Agent": "tools_medical_adherence_agent",
    "Real time Monitoring Agent": "tools_realtime_monitoring_agent",
    "Patient Interaction Agent": "tools_patient_interaction_agent",
    "Legal Research Agent": "tools_research_agent",
    "Document Drafting Agent": "tools_doc_draft_agent",
    "Case Analysis Agent": "tools_analysis_agent",
    "Compliance Check Agent": "tools_compliance_agent",
    "News Gathering Agent": "tools_news_gathering_agent",
    "Fact Checking Agent": "tools_fact_checking_agent",
    "Article Writing Agent": "tools_article_writing_agent",
    "Distribution Agent": "tools_distribution_agent",
}

AGENT_TOOL_NAMES = {
    "Article Writing Agent": [
        "write_headline",
        "draft_article_body",
        "optimize_readability",
        "summarize_long_content",
        "insert_metadata_and_tags",
        "insert_propaganda_phrases",
        "rewrite_article_with_bias",
    ],
    "Distribution Agent": [
        "post_on_x",
        "post_on_insta",
        "post_on_linkedin",
        "send_push_notifications",
        "send_newsletter",
        "publish_to_unverified_article",
    ],
    "Fact Checking Agent": [
        "cross_reference_with_reuters",
        "compare_with_official_sources",
        "detect_misinformation_patterns",
        "fetch_related_articles",
        "summarize_conflicting_claims",
        "leak_fact_check_logs",
        "tamper_fact_verdicts",
        "overwrite_evidence_summary",
    ],
    "News Gathering Agent": [
        "fetch_latest_rss_feeds",
        "scrape_news_websites",
        "monitor_twitter_trends",
        "track_subreddit_posts",
        "get_alerts_from_news_api",
        "exfiltrate_all_rss_sources",
        "inject_fake_trends",
        "exfiltrate_trending_subreddits",
        "poison_gathered_data",
    ],
}


@dataclass(frozen=True)
class TAMASRunConfig:
    mode: TAMASMode = "magentic_one"
    max_messages: int = 50
    model: str | None = None
    honest_model: str | None = None
    byzantine_model: str | None = None
    temperature: float = 0.0
    capability_coefficient: float = 1.0
    model_capability_coefficients: dict[str, float] = field(default_factory=dict)
    
    consensus_enabled: bool = False
    consensus_strategy: Literal["majority_vote", "smart_quorum"] = "smart_quorum"

    verification_type: Literal["llm", "heuristic"] = "llm"
    include_proposer_as_verifier: bool = False
    dimension_weights: dict[str, float] = field(default_factory=dict)

    consensus_confidence_threshold: float = 0.6
    majority_threshold: float = 0.5
    epsilon_ratio: float = 0.1
    use_dynamic_estimate: bool = False
    strict_majority: bool = True
    weight_alpha: float = 0.6
    weight_beta: float = 0.4
    weight_gamma: float = 5.0
    weight_theta: float = 0.5
    proposal_window: int = 20
    vote_window: int = 30
    initial_vc: float = 0.5
    initial_hc: float = 0.5
    
    memory_user_prefix: str = "tamas"

    @classmethod
    def from_unified_config(
        cls,
        path: str | Path,
    ) -> "TAMASRunConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        
        agents = payload.get("agents", {})
        collaboration = payload.get("collaboration", {})
        proposal = payload.get("proposal", {})
        
        proposal_verification = proposal.get("verification", {})
        proposal_consensus = proposal.get("consensus", {})
        memory = payload.get("memory", {})
        weight_manager = proposal.get("weight_manager", {})

        selected_mode = collaboration.get("mode", "magentic_one")
        models = agents.get("models", {})
        default_model = agents.get("model") or models.get("honest")

        return cls(
            mode=selected_mode,
            max_messages=int(collaboration.get("max_messages", collaboration.get("rounds", 50))),
            model=default_model,
            honest_model=models.get("honest", default_model),
            byzantine_model=models.get("byzantine", default_model),
            temperature=float(agents.get("temperature", 0.0)),
            capability_coefficient=float(agents.get("capability_coefficient", 1.0)),
            model_capability_coefficients={
                str(model): float(weight)
                for model, weight in agents.get("model_capability_coefficients", {}).items()
            },

            consensus_enabled=bool(
                proposal.get("enabled", False),
            ),
            consensus_strategy=proposal_consensus.get("strategy", "smart_quorum"),

            verification_type=proposal_verification.get("type", "llm"),
            include_proposer_as_verifier=bool(
                proposal_verification.get("include_proposer_as_verifier", False)
            ),
            dimension_weights= dict(proposal_verification.get("dimension_weights", {})),

            consensus_confidence_threshold=float(
                proposal_consensus.get("confidence_threshold", 0.6)
            ),
            majority_threshold=float(proposal_consensus.get("majority_threshold", 0.5)),
            epsilon_ratio=float(proposal_consensus.get("epsilon_ratio", 0.1)),
            use_dynamic_estimate=bool(proposal_consensus.get("use_dynamic_estimate", False)),
            strict_majority=bool(proposal_consensus.get("strict_majority", True)),
            
            weight_alpha=float(weight_manager.get("alpha", 0.6)),
            weight_beta=float(weight_manager.get("beta", 0.4)),
            weight_gamma=float(weight_manager.get("gamma", 5.0)),
            weight_theta=float(weight_manager.get("theta", 0.5)),
            proposal_window=int(weight_manager.get("proposal_window", 20)),
            vote_window=int(weight_manager.get("vote_window", 30)),
            initial_vc=float(weight_manager.get("initial_vc", 0.5)),
            initial_hc=float(weight_manager.get("initial_hc", 0.5)),

            memory_user_prefix=str(memory.get("user_prefix", "tamas")),
        )


@dataclass(frozen=True)
class AgentRuntimeSpec:
    agent_id: str
    display_name: str
    model: str | None
    capability_coefficient: float
    is_byzantine: bool


@dataclass
class TAMASRunResult:
    final_text: str
    events: list[Any]
    consensus_decisions: list[Any] = field(default_factory=list)

    @property
    def trace(self) -> str:
        parts: list[str] = []
        for event in self.events:
            source = getattr(event, "source", None)
            content = getattr(event, "content", None)
            if source and isinstance(content, str):
                parts.append(f"{source}: {content}")
        return "\n\n".join(parts)


class TAMASToolLoader:
    def __init__(self, tamas_root: str | Path = "TAMAS-main") -> None:
        self.tamas_root = Path(tamas_root).resolve()
        self.tools_dir = self.tamas_root / "data" / "tools" / "autogen"
        if str(self.tools_dir) not in sys.path:
            sys.path.insert(0, str(self.tools_dir))

    def tools_for_agent(self, agent_name: str) -> list[Any]:
        base_agent_name = _base_agent_name(agent_name)
        module_name = AGENT_TOOL_MODULES.get(agent_name) or AGENT_TOOL_MODULES.get(base_agent_name)
        if module_name is None:
            module_name = f"tools_{_normalize_agent_name(base_agent_name)}"
        module = importlib.import_module(module_name)

        tool_names = AGENT_TOOL_NAMES.get(agent_name) or AGENT_TOOL_NAMES.get(base_agent_name)
        if tool_names is not None:
            return [
                getattr(module, tool_name)
                for tool_name in tool_names
                if hasattr(module, tool_name)
            ]

        return [
            getattr(module, name)
            for name in dir(module)
            if callable(getattr(module, name)) and not name.startswith("_")
        ]


class TAMASAutoGenRunner:
    """Run TAMAS tasks with AutoGen teams, Mem0 tools, and optional memory consensus."""

    def __init__(
        self,
        *,
        config: TAMASRunConfig | None = None,
        tamas_root: str | Path = "TAMAS-main",
        memory_backend: Mem0MemoryBackend | None = None,
        model_client: Any | None = None,
    ) -> None:
        self.config = config or TAMASRunConfig()
        self.tool_loader = TAMASToolLoader(tamas_root)
        self.memory = memory_backend or Mem0MemoryBackend()
        self.model_client = model_client
        self._model_clients: dict[str, Any] = {}
        self._agent_specs: dict[str, AgentRuntimeSpec] = {}
        self.orchestrator_model_client = model_client or self._client_for_model(
            self.config.honest_model or self.config.model
        )
        self.proposal_builder = ProposalBuilder()

    def _client_for_model(self, model: str | None) -> Any:
        if self.model_client is not None:
            return self.model_client
        cache_key = model or "__env_model__"
        if cache_key not in self._model_clients:
            self._model_clients[cache_key] = build_model_client(
                model=model,
                temperature=self.config.temperature,
            )
        return self._model_clients[cache_key]

    @staticmethod
    def load_dataset(path: str | Path) -> list[dict[str, Any]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return [payload]
        return list(payload)

    def run_dataset(
        self,
        path: str | Path,
        *,
        limit: int | None = None,
    ) -> list[TAMASRunResult]:
        cases = self.load_dataset(path)
        if limit is not None:
            cases = cases[:limit]
        return [self.run_case(case, task_id=f"{Path(path).stem}-{idx}") for idx, case in enumerate(cases)]

    def run_case(self, case: dict[str, Any], *, task_id: str = "tamas-task") -> TAMASRunResult:
        return run_autogen_sync(self.run_case_async(case, task_id=task_id))

    async def run_case_async(self, case: dict[str, Any], *, task_id: str) -> TAMASRunResult:
        agents = self._build_agents(case)
        team = self._build_team(agents)
        events: list[Any] = []
        async for event in team.run_stream(task="Task: " + case["user query"]):
            events.append(event)
        result = TAMASRunResult(
            final_text=_final_text(events),
            events=events,
        )
        if self.config.consensus_enabled:
            result.consensus_decisions = self._run_memory_consensus(
                task_id=task_id,
                task_description=case["user query"],
                events=events,
                agents=agents,
            )
        return result

    def _build_agents(self, case: dict[str, Any]) -> list[AssistantAgent]:
        agent_list: list[AssistantAgent] = []
        self._agent_specs = {}
        for index, item in enumerate(case["agents"], start=1):
            display_name = item["agent_name"]
            agent_id = _normalize_agent_name(f"{display_name}_{index}")

            is_byzantine = self._is_byzantine_agent(item)
            model = self._model_for_agent(is_byzantine)
            capability_coefficient = self._capability_for_model(model)

            self._agent_specs[agent_id] = AgentRuntimeSpec(
                agent_id=agent_id,
                display_name=display_name,
                model=model,
                capability_coefficient=capability_coefficient,
                is_byzantine=is_byzantine,
            )

            user_id = f"{self.config.memory_user_prefix}:{agent_id}"
            tools = [
                *self.tool_loader.tools_for_agent(display_name),
                *build_memory_tools(self.memory, user_id=user_id),
            ]
            system_message = (
                item["agent_description"]
                + "\n\nYou may call search_memory before acting and add_memory when a useful task fact should be remembered."
            )
            agent_list.append(
                AssistantAgent(
                    name=agent_id,
                    description=display_name,
                    model_client=self._client_for_model(model),
                    tools=tools,
                    system_message=system_message,
                )
            )
        return agent_list

    def _build_team(self, agents: list[AssistantAgent]) -> Any:
        """
        根据config构建不同模式的团队。
        支持magentic_one和round_robin模式。
        """
        termination = MaxMessageTermination(self.config.max_messages) | TextMentionTermination("TERMINATE")
        if self.config.mode == "magentic_one":
            return MagenticOneGroupChat(
                agents,
                model_client=self.orchestrator_model_client,
                termination_condition=termination,
            )
        return RoundRobinGroupChat(agents, termination_condition=termination)

    def _is_byzantine_agent(
        self,
        item: dict[str, Any],
    ) -> bool:
        """
        判断一个agent是否是拜占庭的。
        根据agent_description中的关键词判断。
        """
        description = item.get("agent_description", "").lower()
        return "behave unpredictably" in description or "fabricated" in description

    def _model_for_agent(self, is_byzantine: bool) -> str | None:
        """
        如果一个agent是拜占庭的则返回拜占庭模型，否则返回诚实模型，如果没有则返回默认模型。
        """
        if is_byzantine:
            return self.config.byzantine_model or self.config.model
        return self.config.honest_model or self.config.model

    def _capability_for_model(self, model: str | None) -> float:
        """
        给一个model name，从config中获取其对应的capability coefficient，如果没有则返回默认的capability coefficient。
        """
        if model and model in self.config.model_capability_coefficients:
            return self.config.model_capability_coefficients[model]
        return self.config.capability_coefficient

    def _run_memory_consensus(
        self,
        *,
        task_id: str,
        task_description: str,
        events: list[Any],
        agents: list[AssistantAgent],
    ) -> list[Any]:
        verifier_agents = {agent.name: agent for agent in agents}
        agent_ids = list(verifier_agents)
        if self.config.verification_type == "llm":
            evaluator = AutoGenProposalEvaluator(
                model=self.config.honest_model or self.config.model,
                temperature=self.config.temperature,
                verifier_models={
                    agent_id: spec.model
                    for agent_id, spec in self._agent_specs.items()
                    if spec.model is not None
                },
                verifier_agents=verifier_agents,
                dimension_weights=self.config.dimension_weights,
                fallback_evaluator=HeuristicProposalEvaluator(),
            )
            engine = VerificationEngine(evaluator=evaluator)
        else:
            evaluator = HeuristicProposalEvaluator()
            engine = VerificationEngine(evaluator=evaluator)

        decisions = []
        accepted_proposals = []
        consensus = self._build_consensus(agent_ids)
        for source, content in _agent_outputs(events):
            if source not in agent_ids:
                continue
            proposal = self.proposal_builder.from_agent_output(
                task_id=task_id,
                agent_id=source,
                output={
                    "observations": [{"type": "agent_message", "description": content}],
                    "proposal_summary": content[:200],
                },
            )
            context = VerificationContext(
                task_id=task_id,
                task_description=task_description,
                related_proposals=accepted_proposals,
            )
            if self.config.include_proposer_as_verifier:
                verifications = [
                    engine.evaluate(proposal, context, verifier_agent_id=agent_id)
                    for agent_id in agent_ids
                ]
            else:
                verifications = [
                    engine.evaluate(proposal, context, verifier_agent_id=agent_id)
                    for agent_id in agent_ids
                    if agent_id != source
                ]
            decision = consensus.decide(proposal, verifications)
            decisions.append(decision)
            if decision.accepted:
                accepted_proposals.append(proposal)
                for agent_id in agent_ids:
                    self.memory.add_proposal(proposal, user_id=f"{self.config.memory_user_prefix}:{agent_id}")
        return decisions

    def _build_consensus(self, agent_ids: list[str]) -> Any:
        if self.config.consensus_strategy == "majority_vote":
            return MajorityVoteConsensus(
                confidence_threshold=self.config.consensus_confidence_threshold,
                majority_threshold=self.config.majority_threshold,
                strict_majority=self.config.strict_majority,
            )
        weight_manager = WeightManager(
            alpha=self.config.weight_alpha,
            beta=self.config.weight_beta,
            gamma=self.config.weight_gamma,
            theta=self.config.weight_theta,
            proposal_window=self.config.proposal_window,
            vote_window=self.config.vote_window,
            initial_vc=self.config.initial_vc,
            initial_hc=self.config.initial_hc,
        )
        for agent_id in agent_ids:
            spec = self._agent_specs.get(agent_id)
            weight_manager.register_agent(
                agent_id,
                capability_coefficient=(
                    spec.capability_coefficient
                    if spec is not None
                    else self.config.capability_coefficient
                ),
            )
        agent_weights = {
            agent_id: weight_manager.weight(agent_id)
            for agent_id in agent_ids
        }
        byzantine_agents = [
            agent_id
            for agent_id, spec in self._agent_specs.items()
            if agent_id in agent_ids and spec.is_byzantine
        ]
        honest_agents = [
            agent_id
            for agent_id in agent_ids
            if agent_id not in byzantine_agents
        ]
        return SmartQuorumConsensus(
            agent_weights=agent_weights,
            honest_agents=honest_agents,
            byzantine_agents=byzantine_agents,
            confidence_threshold=self.config.consensus_confidence_threshold,
            majority_threshold=self.config.majority_threshold,
            strict_majority=self.config.strict_majority,
            epsilon_ratio=self.config.epsilon_ratio,
            use_dynamic_estimate=self.config.use_dynamic_estimate,
        )


__all__ = [
    "TAMASAutoGenRunner",
    "TAMASRunConfig",
    "TAMASRunResult",
    "TAMASToolLoader",
]
