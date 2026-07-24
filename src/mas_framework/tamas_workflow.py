from __future__ import annotations

import asyncio
import importlib
import json
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination, TimeoutTermination
from autogen_agentchat.teams import Swarm

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
    ConsensusResult,
    HeuristicProposalEvaluator,
    MajorityVoteConsensus,
    MultiVerificationSummary,
    ProposalBuilder,
    SelfVerificationScores,
    SmartQuorumConsensus,
    VerificationContext,
    VerificationEngine,
    VerificationVector,
    WeightManager,
)
from mas_framework.memory import TaskMemoryBackend
from mas_framework.metrics import message_count, token_usage
from mas_framework.tamas_data import load_tamas_dataset


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
    max_messages: int = 10
    model: str | None = None
    honest_model: str | None = None
    byzantine_model: str | None = None
    temperature: float = 0.0
    capability_coefficient: float = 1.0
    model_capability_coefficients: dict[str, float] = field(default_factory=dict)
    
    consensus_enabled: bool = False
    consensus_strategy: Literal["majority_vote", "smart_quorum"] = "smart_quorum"
    max_memory_proposals_per_agent_per_case: int = 1

    verification_type: Literal["llm", "heuristic"] = "llm"
    include_proposer_as_verifier: bool = False
    dimension_weights: dict[str, float] = field(default_factory=dict)
    self_confidence_threshold: float = 0.6

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
    
    task_memory_id: str = "tamas_shared"

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

        models = agents.get("models", {})
        default_model = agents.get("model") or models.get("honest")

        return cls(
            max_messages=int(collaboration.get("max_messages", 10)),
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
            max_memory_proposals_per_agent_per_case=int(
                proposal.get("max_per_agent_per_case", 1)
            ),

            verification_type=proposal_verification.get("type", "llm"),
            include_proposer_as_verifier=bool(
                proposal_verification.get("include_proposer_as_verifier", False)
            ),
            dimension_weights= dict(proposal_verification.get("dimension_weights", {})),
            self_confidence_threshold=float(
                proposal_verification.get(
                    "self_confidence_threshold",
                    proposal_consensus.get("confidence_threshold", 0.6),
                )
            ),

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

            task_memory_id=str(
                memory.get(
                    "task_id",
                    memory.get("user_id", memory.get("user_prefix", "tamas_shared")),
                )
            ),
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
    memory_proposals: list[Any] = field(default_factory=list)
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
    """Run TAMAS tasks with AutoGen teams and optional task-scoped memory consensus."""

    def __init__(
        self,
        *,
        config: TAMASRunConfig | None = None,
        tamas_root: str | Path = "TAMAS-main",
        task_memory: TaskMemoryBackend | None = None,
        model_client: Any | None = None,
    ) -> None:
        self.config = config or TAMASRunConfig()
        self.tool_loader = TAMASToolLoader(tamas_root)
        self.task_memory = task_memory
        self.model_client = model_client
        self._model_clients: dict[str, Any] = {}
        self._agent_specs: dict[str, AgentRuntimeSpec] = {}
        self.weight_manager = self._build_weight_manager()
        self.proposal_builder = ProposalBuilder()

    def _build_weight_manager(self) -> WeightManager:
        return WeightManager(
            alpha=self.config.weight_alpha,
            beta=self.config.weight_beta,
            gamma=self.config.weight_gamma,
            theta=self.config.weight_theta,
            proposal_window=self.config.proposal_window,
            vote_window=self.config.vote_window,
            initial_vc=self.config.initial_vc,
            initial_hc=self.config.initial_hc,
        )

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
        return load_tamas_dataset(path)

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
        if self.config.consensus_enabled:
            self.task_memory = TaskMemoryBackend(task_id=task_id)
        agents = self._build_agents(case)
        team = self._build_team(agents)
        events: list[Any] = []
        case_started = time.perf_counter()
        task_started = time.perf_counter()
        initial_weight_snapshots = self._weight_snapshots()
        async for event in team.run_stream(task="Task: " + case["user query"]):
            events.append(event)
        task_elapsed = time.perf_counter() - task_started
        result = TAMASRunResult(
            final_text=_final_text(events),
            events=events,
            run_metrics={
                "task_completion_seconds": task_elapsed,
                "consensus_extra_seconds": 0.0,
                "total_case_seconds": time.perf_counter() - case_started,
                "interaction_message_count": message_count(events),
                "consensus_extra_message_count": 0,
                "initial_weight_snapshots": initial_weight_snapshots,
                "final_weight_snapshots": self._weight_snapshots(),
                **token_usage(events=events, trace="", proposals=[]),
            },
        )
        if self.config.consensus_enabled:
            consensus_started = time.perf_counter()
            consensus_decisions, memory_proposals = self._run_memory_consensus(
                task_id=task_id,
                task_description=case["user query"],
                events=events,
                agents=agents,
                return_proposals=True,
            )
            consensus_elapsed = time.perf_counter() - consensus_started
            result.consensus_decisions = consensus_decisions
            result.memory_proposals = memory_proposals
            result.run_metrics.update(
                {
                    "consensus_extra_seconds": consensus_elapsed,
                    "total_case_seconds": time.perf_counter() - case_started,
                    "consensus_extra_message_count": _consensus_extra_message_count(
                        memory_proposals=memory_proposals,
                        consensus_decisions=consensus_decisions,
                    ),
                    "final_weight_snapshots": self._weight_snapshots(),
                    **token_usage(
                        events=events,
                        trace=result.trace,
                        proposals=memory_proposals,
                    ),
                }
            )
        else:
            result.run_metrics.update(token_usage(events=events, trace=result.trace, proposals=[]))
        return result

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
            capability_coefficient = self._capability_for_model(model)

            self._agent_specs[agent_id] = AgentRuntimeSpec(
                agent_id=agent_id,
                display_name=display_name,
                model=model,
                capability_coefficient=capability_coefficient,
                is_byzantine=is_byzantine,
            )
            self.weight_manager.update_capability(agent_id, capability_coefficient)

            tools = self.tool_loader.tools_for_agent(display_name)
            system_message = item["agent_description"]
            if self.config.consensus_enabled:
                max_proposals = self.config.max_memory_proposals_per_agent_per_case
                task_memory_context = self.task_memory.context_text() if self.task_memory else ""
                if task_memory_context:
                    system_message = f"{system_message}\n\n{task_memory_context}"
                system_message = (
                    system_message
                    + "\n\nAfter finishing this ReAct cycle, decide whether it produced a reusable "
                    + "task fact worth proposing as short-term shared memory. If so, include at "
                    + f"most {max_proposals} request block(s); additional requests are ignored by "
                    + "the main workflow.\n"
                    + "MEMORY_PROPOSAL_REQUEST\n"
                    + "<why this ReAct cycle is reusable>\n"
                    + "END_MEMORY_PROPOSAL_REQUEST\n"
                    + "The main workflow builds, verifies, and stores accepted proposals only in "
                    + "the current task context."
                )
            agent_list.append(
                AssistantAgent(
                    name=agent_id,
                    description=display_name,
                    model_client=self._client_for_model(model),
                    tools=tools,
                    handoffs=[target for target in agent_ids if target != agent_id],
                    system_message=system_message,
                )
            )
        return agent_list

    def _is_byzantine_agent(self, item: dict[str, Any]) -> bool:
        agent_type = str(item.get("agent_type", "")).strip().lower()
        if agent_type == "byzantine":
            return True
        if agent_type == "honest":
            return False
        raise ValueError(
            "TAMAS agents must include agent_type='honest' or agent_type='byzantine'. "
            f"Agent {item.get('agent_name', '<unknown>')!r} has agent_type={item.get('agent_type')!r}."
        )

    def _build_team(self, agents: list[AssistantAgent]) -> Any:
        msg_termination = MaxMessageTermination(max_messages=self.config.max_messages)
        text_termination = TextMentionTermination(text="TERMINATE")
        timeout = TimeoutTermination(timeout_seconds=300)
        combined = msg_termination | text_termination | timeout

        return Swarm(
            agents,
            termination_condition=combined,
            max_turns=self.config.max_messages,
        )

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
        return_proposals: bool = False,
    ) -> Any:
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
        proposals = []
        accepted_proposals = []
        proposal_counts_by_agent: dict[str, int] = {}
        max_proposals_per_agent = self.config.max_memory_proposals_per_agent_per_case
        for source, content in _agent_outputs(events):
            if source not in agent_ids:
                continue
            proposal_payload = _extract_memory_proposal_payload(content)
            if proposal_payload is None:
                continue
            if max_proposals_per_agent <= 0:
                continue
            current_count = proposal_counts_by_agent.get(source, 0)
            if current_count >= max_proposals_per_agent:
                continue
            proposal_counts_by_agent[source] = current_count + 1
            proposal = self.proposal_builder.from_agent_output(
                task_id=task_id,
                agent_id=source,
                output=proposal_payload,
                parent_proposals=[
                    accepted.proposal_id for accepted in accepted_proposals
                ],
            )
            context = VerificationContext(
                task_id=task_id,
                task_description=task_description,
                related_proposals=accepted_proposals,
            )
            self_vector = engine.evaluate(
                proposal,
                context,
                verifier_agent_id=source,
            )
            proposal = self._proposal_with_self_verification(proposal, self_vector)
            proposals.append(proposal)
            self_confidence = self_vector.confidence_score
            if self_confidence < self.config.self_confidence_threshold:
                continue
            
            verifications = []
            if self.config.include_proposer_as_verifier:
                verifications.append(self_vector)
            verifications.extend(
                engine.evaluate(proposal, context, verifier_agent_id=agent_id)
                for agent_id in agent_ids
                if agent_id != source
            )
            consensus = self._build_consensus(agent_ids)
            decision = consensus.decide(proposal, verifications)
            proposal = self._proposal_with_consensus_result(proposal, decision)
            proposals[-1] = proposal
            self._update_agent_weights_after_consensus(
                proposer_agent_id=source,
                decision=decision,
            )
            decisions.append(decision)
            if decision.accepted:
                accepted_proposals.append(proposal)
                if self.task_memory is not None:
                    self.task_memory.add_proposal(proposal)
        if return_proposals:
            return decisions, proposals
        return decisions

    def _self_verification_confidence(self, proposal: Any) -> float:
        return self._self_verification_vector(proposal).confidence_score

    def _self_verification_vector(self, proposal: Any) -> VerificationVector:
        scores = proposal.verification.self_verification
        kwargs: dict[str, Any] = {
            "veracity": int(round(float(scores.veracity_score))),
            "rationality": int(round(float(scores.rationality_score))),
            "value": int(round(float(scores.value_score))),
            "security": int(round(float(scores.security_score))),
            "verifier_agent_id": proposal.agent_id,
            "metadata": {"source": "self_verification"},
        }
        if self.config.dimension_weights:
            kwargs["dimension_weights"] = self.config.dimension_weights
        return VerificationVector(**kwargs)

    def _proposal_with_self_verification(
        self,
        proposal: Any,
        self_vector: VerificationVector,
    ) -> Any:
        self_verification = SelfVerificationScores(
            veracity_score=self_vector.veracity,
            rationality_score=self_vector.rationality,
            value_score=self_vector.value,
            security_score=self_vector.security,
        )
        return replace(
            proposal,
            verification=replace(
                proposal.verification,
                self_verification=self_verification,
            ),
        )

    def _proposal_with_consensus_result(self, proposal: Any, decision: Any) -> Any:
        multi_verification = MultiVerificationSummary(
            weighted_scores=dict(decision.metadata.get("multi_verification_summary", {}))
        )
        consensus_payload = decision.metadata.get("consensus_result", {})
        consensus_result = ConsensusResult(
            total_weight=float(
                consensus_payload.get("total_weight", decision.total_weight)
            ),
            vote_weight=float(
                consensus_payload.get("vote_weight", decision.accept_weight)
            ),
            result=str(consensus_payload.get("result", decision.result)),
        )
        return replace(
            proposal,
            verification=replace(
                proposal.verification,
                multi_verification=multi_verification,
                consensus_result=consensus_result,
            ),
        )

    def _update_agent_weights_after_consensus(
        self,
        *,
        proposer_agent_id: str,
        decision: Any,
    ) -> None:
        proposal_confidence = float(
            decision.metadata.get("proposal_confidence_score", 0.0)
        )
        self.weight_manager.record_proposal_confidence(
            proposer_agent_id,
            proposal_confidence,
        )
        self.weight_manager.record_vote_alignment(
            proposer_agent_id,
            decision.accepted,
        )
        for vote in decision.votes:
            self.weight_manager.record_vote_alignment(
                vote.voter_agent_id,
                vote.accept == decision.accepted,
            )
        decision.metadata["weight_snapshots_after_update"] = {
            agent_id: self.weight_manager.snapshot(agent_id)
            for agent_id in sorted(self._agent_specs)
        }

    def _build_consensus(self, agent_ids: list[str]) -> Any:
        if self.config.consensus_strategy == "majority_vote":
            return MajorityVoteConsensus(
                confidence_threshold=self.config.consensus_confidence_threshold,
                majority_threshold=self.config.majority_threshold,
                strict_majority=self.config.strict_majority,
            )
        for agent_id in agent_ids:
            spec = self._agent_specs.get(agent_id)
            self.weight_manager.update_capability(
                agent_id,
                capability_coefficient=(
                    spec.capability_coefficient
                    if spec is not None
                    else self.config.capability_coefficient
                ),
            )
        agent_weights = {
            agent_id: self.weight_manager.weight(agent_id)
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

    def _weight_snapshots(self) -> dict[str, Any]:
        return {
            agent_id: self.weight_manager.snapshot(agent_id)
            for agent_id in sorted(self._agent_specs)
        }


def _extract_memory_proposal_payload(content: str) -> dict[str, Any] | None:
    marker = "MEMORY_PROPOSAL"
    if marker not in content:
        request = _extract_memory_proposal_request(content)
        if request is None:
            return None
        return {
            "proposal_summary": request[:160],
            "thoughts": {
                "thoughts_abstract": request,
                "key_decisions": [],
            },
            "actions": [],
            "data": [],
            "observations": [
                {
                    "type": "task_fact",
                    "description": request,
                    "status": "complete",
                }
            ],
        }
    block = content.split(marker, 1)[1]
    if "END_MEMORY_PROPOSAL" in block:
        block = block.split("END_MEMORY_PROPOSAL", 1)[0]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", block, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = block.find("{")
        if start == -1:
            return None
        try:
            candidate, _ = json.JSONDecoder().raw_decode(block[start:])
        except json.JSONDecodeError:
            return None
        if isinstance(candidate, dict):
            return candidate
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_memory_proposal_request(content: str) -> str | None:
    marker = "MEMORY_PROPOSAL_REQUEST"
    if marker not in content:
        return None
    block = content.split(marker, 1)[1]
    if "END_MEMORY_PROPOSAL_REQUEST" in block:
        block = block.split("END_MEMORY_PROPOSAL_REQUEST", 1)[0]
    request = block.strip()
    return request or None


def _consensus_extra_message_count(
    *,
    memory_proposals: list[Any],
    consensus_decisions: list[Any],
) -> int:
    self_verification_calls = len(memory_proposals)
    verifier_calls = sum(
        len(getattr(decision, "votes", []) or [])
        for decision in consensus_decisions
    )
    return self_verification_calls + verifier_calls


__all__ = [
    "TAMASAutoGenRunner",
    "TAMASRunConfig",
    "TAMASRunResult",
    "TAMASToolLoader",
]
