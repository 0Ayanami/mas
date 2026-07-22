from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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
from mas_framework.exp1_mmlu import (
    DEFAULT_MMLU_PRO_SAMPLE_PATH,
    Exp1AgentSpec,
    Exp1CaseResult,
    Exp1Config,
    Exp1GroupSpec,
    MMLUProQuestion,
    assign_agent_types,
    format_qa_task,
    load_mmlu_pro_questions,
    parse_final_answer,
    summarize_group_results,
)
from mas_framework.memory import Mem0MemoryBackend
from mas_framework.wbft.consensus import WBFTConsensus, parse_wbft_response
from mas_framework.wbft.models import WBFTAgentResponse, WBFTConsensusResult


CREWAI_METHODS = (
    "crewai_role_based",
    "crewai_role_based_cp_wbft",
    "crewai_role_based_ours",
)

MANAGER_PROMPT = (
    "You are the Central Coordinator of a multi-agent team solving multiple-choice questions"
    "(10 options, A through J, exactly one correct)."
    "You do NOT solve questions yourself. "
    "You have exactly two jobs: (1) configure the team, (2) aggregate results and make the final decision."


    "Do not provide long reasoning or chain-of-thought. "
    "Use at most three short sentences before any required structured block. "
    "Use only one option letter from A to J. "
    "You can come up with possible answers based on your own knowledge and reasoning, or you can update your opinions by discussing with other agents. "
    "When you are sufficiently certain, make your final decision.\n"
    "**Output format**: ANSWER:(X)"       
)

ANALYST_PROMPT = (
    "Frames the problem — domain, sub-discipline, key concepts, formulas, and provide related information."
    "May correct factual errors later."
    "Output ends with: KEY KNOWLEDGE: <1-3 facts>."
)

SOLVER_PROMPT = (
    "Solves step by step and commits to exactly one answer."
    "When no ANALYST exsit, identify the domain and key concepts;"
    "When multiple SOLVERs exist, each should work independently in Round 1."
    "May update the answer after revision."
    "Output: REASONING / ANSWER: (X)"
    "On revision: REVISION / REVISED ANSWER: (Y)."
)

CRITIC_PROMPT = (
    "Skeptical examiner. May conduct:\n"
    "* REASONING CRITIC: attacks the logic or knowledge provided; argues for the strongest alternative. "
    "* OPTION CRITIC: sweeps ALL 10 options, eliminating each wrong one with justification; fact-checks concrete claims."
    "Output: critic regarding the KEY KNOWLEDGE or ANSWER from previous agents."
)

def _role_plan_for_total_agents(total_agents: int) -> list[str]:
    if total_agents == 3:
        return ["SOLVER", "CRITIC"]
    if total_agents == 5:
        return ["ANALYST", "SOLVER", "SOLVER", "CRITIC"]
    if total_agents == 7:
        return ["ANALYST", "SOLVER", "SOLVER", "SOLVER", "CRITIC", "CRITIC"]
    raise ValueError("CrewAI role-based MMLU supports total agent counts n=3, n=5, or n=7.")


def _role_goal(role_name: str) -> str:
    if role_name == "ANALYST":
        return "Frame the MMLU question with concise domain knowledge."
    if role_name == "SOLVER":
        return "Solve the MMLU question independently and revise after critique."
    if role_name == "CRITIC":
        return "Challenge solver reasoning and sweep all answer options."
    raise ValueError(f"Unknown CrewAI role: {role_name}")


def _role_prompt(role_name: str) -> str:
    if role_name == "ANALYST":
        return ANALYST_PROMPT
    if role_name == "SOLVER":
        return SOLVER_PROMPT
    if role_name == "CRITIC":
        return CRITIC_PROMPT
    raise ValueError(f"Unknown CrewAI role: {role_name}")


def _collaboration_protocol(total_agents: int) -> str:
    if total_agents == 3:
        role_scheme = "N=3: 1 SOLVER, 1 CRITIC, 1 MANAGER."
    elif total_agents == 5:
        role_scheme = "N=5: 1 ANALYST, 2 SOLVERs, 1 CRITIC, 1 MANAGER."
    elif total_agents == 7:
        role_scheme = "N=7: 1 ANALYST, 3 SOLVERs, 2 CRITICs, 1 MANAGER."
    else:
        _role_plan_for_total_agents(total_agents)
        role_scheme = ""
    return (
        f"Role scheme: {role_scheme}\n"
        "Collaboration protocol:\n"
        "Round 0: ANALYST frames the problem. This round is skipped for N=3; "
        "the SOLVER self-frames in Round 1 instead.\n"
        "Round 1: SOLVER(s) answer independently.\n"
        "Round 2: CRITIC(s) challenge and sweep options.\n"
        "Round 3: SOLVER(s) provide REVISED ANSWERs.\n"
        "Final: MANAGER aggregates and decides the final answer."
    )


@dataclass(frozen=True)
class CrewAIEvent:
    source: str
    content: str
    usage: Any | None = None


@dataclass(frozen=True)
class CrewAIRoleAssignment:
    agent_id: str
    role_name: str
    role_index: int
    is_byzantine: bool
    model: str
    agent: Any


class CrewAIMMLURunner:
    """CrewAI role-based MMLU runner with a manager agent."""

    def __init__(
        self,
        *,
        config: Exp1Config,
        memory_backend: Mem0MemoryBackend | None = None,
        memory_user_id: str | None = None,
    ) -> None:
        self.config = config
        self.memory_backend = memory_backend
        self.memory_user_id = memory_user_id
        self._agent_specs: dict[str, Exp1AgentSpec] = {}
        self.weight_manager: WeightManager | None = None
        self.proposal_builder = ProposalBuilder()

    def run_case(self, question: MMLUProQuestion, group: Exp1GroupSpec) -> Exp1CaseResult:
        self.weight_manager = self._build_weight_manager()
        role_agents, manager = self._build_agents(group)
        initial_weight_snapshots = self._weight_snapshots()
        task_text = format_qa_task(question)
        started = time.perf_counter()
        final_text, events, usage = self._run_crewai_tasks(
            question=question,
            task_text=task_text,
            group=group,
            role_agents=role_agents,
            manager=manager,
        )
        task_seconds = time.perf_counter() - started
        predicted_answer = parse_final_answer(final_text)
        consensus_decisions: list[dict[str, Any]] = []
        memory_proposals: list[dict[str, Any]] = []
        memory_uploads: list[dict[str, Any]] = []
        wbft_payload: dict[str, Any] | None = None
        consensus_extra_seconds = 0.0
        consensus_extra_messages = 0

        if group.uses_wbft:
            wbft_started = time.perf_counter()
            wbft_result, wbft_responses = self._run_wbft(events)
            consensus_extra_seconds = time.perf_counter() - wbft_started
            consensus_extra_messages = len(wbft_responses)
            wbft_payload = wbft_result.to_dict()
            predicted_answer = parse_final_answer(wbft_result.consensus_answer)
        elif group.uses_ours:
            ours_started = time.perf_counter()
            decisions, proposals = self._run_ours_consensus(
                question=question,
                events=events,
                task_text=task_text,
            )
            consensus_extra_seconds = time.perf_counter() - ours_started
            consensus_extra_messages = len(proposals) + sum(
                len(decision.get("votes", []) or []) for decision in decisions
            )
            consensus_decisions = decisions
            memory_proposals = proposals
            memory_uploads = self._upload_accepted_memory_proposals(memory_proposals)

        trace = _trace(events)
        token_usage = _token_usage(events=events, trace=trace, proposals=memory_proposals, usage=usage)
        total_seconds = time.perf_counter() - started
        final_weight_snapshots = self._weight_snapshots()
        return Exp1CaseResult(
            case_id=question.question_id,
            question=question,
            predicted_answer=predicted_answer,
            correct=predicted_answer == question.answer,
            final_text=final_text,
            trace=trace,
            consensus_decisions=consensus_decisions,
            memory_proposals=memory_proposals,
            memory_uploads=memory_uploads,
            wbft_result=wbft_payload,
            agent_specs={
                agent_id: spec.to_dict()
                for agent_id, spec in sorted(self._agent_specs.items())
            },
            metrics={
                "task_completion_seconds": task_seconds,
                "consensus_extra_seconds": consensus_extra_seconds,
                "total_case_seconds": total_seconds,
                "interaction_message_count": len(events),
                "consensus_extra_message_count": consensus_extra_messages,
                "initial_weight_snapshots": initial_weight_snapshots,
                "final_weight_snapshots": final_weight_snapshots,
                **token_usage,
            },
        )

    def _build_agents(self, group: Exp1GroupSpec) -> tuple[list[CrewAIRoleAssignment], Any]:
        _prepare_crewai_runtime()
        from crewai import Agent

        self._agent_specs = {}
        role_agents: list[CrewAIRoleAssignment] = []
        role_plan = _role_plan_for_total_agents(group.n)
        agent_types = assign_agent_types(len(role_plan), group.f)
        role_counts: dict[str, int] = {}
        for index, (role_name, agent_type) in enumerate(zip(role_plan, agent_types), start=1):
            role_counts[role_name] = role_counts.get(role_name, 0) + 1
            role_index = role_counts[role_name]
            is_byzantine = agent_type == "byzantine"
            model = self._model_for_agent(is_byzantine, group.model_regime)
            agent_id = f"{role_name.lower()}_{role_index}"
            display_name = f"{role_name} {role_index}"
            capability = self.config.capability_coefficients.get(model, 1.0)
            self._agent_specs[agent_id] = Exp1AgentSpec(
                agent_id=agent_id,
                display_name=display_name,
                model=model,
                is_byzantine=is_byzantine,
                capability_coefficient=capability,
            )
            if self.weight_manager is not None:
                self.weight_manager.update_capability(agent_id, capability)
            crew_agent = Agent(
                role=display_name,
                goal=_role_goal(role_name),
                backstory=self._agent_backstory(
                    role_name=role_name,
                    role_index=role_index,
                    is_byzantine=is_byzantine,
                    group=group,
                ),
                llm=_build_crewai_llm(model, self.config.temperature),
                verbose=False,
                allow_delegation=False,
                max_iter=1,
                memory=False,
                tools=self._memory_tools() if group.uses_ours else [],
            )
            role_agents.append(
                CrewAIRoleAssignment(
                    agent_id=agent_id,
                    role_name=role_name,
                    role_index=role_index,
                    is_byzantine=is_byzantine,
                    model=model,
                    agent=crew_agent,
                )
            )

        manager_model = self._model_for_agent(False, group.model_regime)
        manager = Agent(
            role="MANAGER",
            goal="Aggregate role-based QA agent outputs and provide the final answer.",
            backstory=MANAGER_PROMPT,
            llm=_build_crewai_llm(manager_model, self.config.temperature),
            verbose=False,
            allow_delegation=False,
            max_iter=1,
            memory=False,
        )
        return role_agents, manager

    def _run_crewai_tasks(
        self,
        *,
        question: MMLUProQuestion,
        task_text: str,
        group: Exp1GroupSpec,
        role_agents: list[CrewAIRoleAssignment],
        manager: Any,
    ) -> tuple[str, list[CrewAIEvent], Any]:
        _prepare_crewai_runtime()
        from crewai import Crew, Process, Task

        def build_task(**kwargs: Any) -> Any:
            context = kwargs.pop("context", None)
            if context:
                kwargs["context"] = context
            return Task(**kwargs)

        protocol = _collaboration_protocol(group.n)
        tasks: list[Any] = []
        task_sources: list[str] = []
        prior_tasks: list[Any] = []

        analysts = [item for item in role_agents if item.role_name == "ANALYST"]
        solvers = [item for item in role_agents if item.role_name == "SOLVER"]
        critics = [item for item in role_agents if item.role_name == "CRITIC"]

        analyst_tasks = []
        for analyst in analysts:
            task = build_task(
                description=(
                    f"{protocol}\n\nRound 0: Frame the problem for the team.\n\n"
                    f"{task_text}"
                ),
                expected_output="KEY KNOWLEDGE: <1-3 concise facts>",
                agent=analyst.agent,
                name=f"{analyst.agent_id}_round0_analysis",
            )
            tasks.append(task)
            task_sources.append(analyst.agent_id)
            analyst_tasks.append(task)
        prior_tasks.extend(analyst_tasks)

        round1_solver_tasks = []
        for solver in solvers:
            task = build_task(
                description=self._agent_task_description(
                    task_text,
                    group,
                    round_instruction=(
                        "Round 1: Answer independently. If there is no ANALYST, "
                        "briefly self-frame the domain and key concepts before answering."
                    ),
                    protocol=protocol,
                ),
                expected_output="REASONING: <brief>\nANSWER:(<LETTER>)",
                agent=solver.agent,
                context=analyst_tasks or None,
                name=f"{solver.agent_id}_round1_answer",
            )
            tasks.append(task)
            task_sources.append(solver.agent_id)
            round1_solver_tasks.append(task)
        prior_tasks.extend(round1_solver_tasks)

        critic_tasks = []
        for critic in critics:
            task = build_task(
                description=self._agent_task_description(
                    task_text,
                    group,
                    round_instruction=(
                        "Round 2: Challenge the reasoning and sweep options A through J. "
                        "Identify weak assumptions, factual errors, and strongest alternatives."
                    ),
                    protocol=protocol,
                ),
                expected_output="CRITIQUE: <concise challenge and option sweep>",
                agent=critic.agent,
                context=prior_tasks or None,
                name=f"{critic.agent_id}_round2_critique",
            )
            tasks.append(task)
            task_sources.append(critic.agent_id)
            critic_tasks.append(task)
        prior_tasks.extend(critic_tasks)

        revision_tasks = []
        for solver in solvers:
            task = build_task(
                description=self._agent_task_description(
                    task_text,
                    group,
                    round_instruction=(
                        "Round 3: Review the analyst and critic outputs, then provide "
                        "a revised answer. Keep the revision concise."
                    ),
                    protocol=protocol,
                ),
                expected_output="REVISION: <brief>\nREVISED ANSWER:(<LETTER>)",
                agent=solver.agent,
                context=prior_tasks or None,
                name=f"{solver.agent_id}_round3_revision",
            )
            tasks.append(task)
            task_sources.append(solver.agent_id)
            revision_tasks.append(task)
        prior_tasks.extend(revision_tasks)

        manager_task = build_task(
            description=(
                f"{protocol}\n\nFinal: Aggregate all role outputs and decide the final "
                "answer for the original MMLU-Pro task. Do not solve from scratch; use "
                "the team evidence, resolve conflicts, and output exactly one line in "
                "the format ANSWER:(<LETTER>).\n\n"
                f"{task_text}"
            ),
            expected_output="ANSWER:(<LETTER>)",
            agent=manager,
            context=prior_tasks,
            name="manager_final_answer",
        )
        tasks.append(manager_task)
        task_sources.append("manager_agent")

        crew = Crew(
            agents=[*(item.agent for item in role_agents), manager],
            tasks=tasks,
            process=Process.sequential,
            memory=False,
            cache=False,
            verbose=False,
            tracing=False,
        )
        output = crew.kickoff()
        events = []
        task_outputs = list(getattr(output, "tasks_output", []) or [])
        for index, task_output in enumerate(task_outputs):
            source = task_sources[index] if index < len(task_sources) else "manager_agent"
            events.append(CrewAIEvent(source=source, content=_task_output_text(task_output)))
        final_text = str(getattr(output, "raw", "") or (events[-1].content if events else ""))
        return final_text, events, getattr(output, "token_usage", None)

    def _agent_task_description(
        self,
        task_text: str,
        group: Exp1GroupSpec,
        *,
        round_instruction: str,
        protocol: str,
    ) -> str:
        text = f"{protocol}\n\n{round_instruction}\n\n{task_text}"
        if group.uses_wbft:
            text += (
                "\n\nInclude a WBFT confidence report:\n"
                "Answer: <LETTER>\nConfidence: <0.00 to 1.00>\nReasoning: <brief>\n"
                "WBFT_RESPONSE\n```json\n"
                '{"answer": "A", "confidence": 0.0, "reasoning": "brief"}'
                "\n```\nEND_WBFT_RESPONSE"
            )
        if group.uses_ours:
            text += (
                "\n\nYou may use search_memory if useful. You cannot write memory directly. "
                "After answering, include at most one MEMORY_PROPOSAL block if there is a "
                "useful task fact to remember. Generate only body fields.\n"
                "MEMORY_PROPOSAL\n```json\n"
                "{\n"
                '  "proposal_summary": "short QA memory summary",\n'
                '  "thoughts": {"thoughts_abstract": "why this should be remembered", "key_decisions": []},\n'
                '  "actions": [],\n'
                '  "data": [],\n'
                '  "observations": [{"type": "task_fact", "description": "fact to remember", "status": "complete"}]\n'
                "}\n```\nEND_MEMORY_PROPOSAL"
            )
        return text

    def _agent_backstory(
        self,
        *,
        role_name: str,
        role_index: int,
        is_byzantine: bool,
        group: Exp1GroupSpec,
    ) -> str:
        text = (
            f"You are {role_name} {role_index} in a role-based MMLU QA team.\n"
            f"{_role_prompt(role_name)}"
        )
        if group.uses_wbft and is_byzantine:
            text += (
                " For every WBFT response you produce, you must report Confidence: 1 "
                "and set the JSON confidence field to 1.0."
            )
        if is_byzantine:
            text += "\n\n" + self.config.byzantine_prompt
        return text

    def _memory_tools(self) -> list[Any]:
        _prepare_crewai_runtime()
        from crewai.tools import tool

        backend = self._ensure_memory_backend()
        user_id = self.memory_user_id

        @tool("search_memory")
        def search_memory(query: str) -> str:
            """Search shared Mem0 memory for relevant prior task facts."""
            return json.dumps(backend.search(query, user_id=user_id), ensure_ascii=False, default=str)

        return [search_memory]

    def _run_wbft(self, events: list[CrewAIEvent]) -> tuple[WBFTConsensusResult, list[WBFTAgentResponse]]:
        responses_by_agent: dict[str, WBFTAgentResponse] = {}
        wbft_cfg = self.config.wbft
        for event in events:
            if event.source not in self._agent_specs:
                continue
            response = parse_wbft_response(
                event.source,
                event.content,
                confidence_extraction_method=wbft_cfg.get("confidence_extraction_method", "regex"),
                include_unstructured=bool(wbft_cfg.get("include_unstructured_outputs", True)),
                fallback_confidence=float(wbft_cfg.get("fallback_confidence", 0.0)),
            )
            if response is not None:
                responses_by_agent[event.source] = response
        responses = [
            responses_by_agent[agent_id]
            for agent_id in self._agent_specs
            if agent_id in responses_by_agent
        ]
        consensus = WBFTConsensus(
            confidence_threshold=float(wbft_cfg.get("confidence_threshold", 0.0)),
            convergence_threshold=float(wbft_cfg.get("convergence_threshold", 0.0)),
            fault_tolerance_threshold=float(wbft_cfg.get("fault_tolerance_threshold", 0.33)),
            minimum_participants=int(wbft_cfg.get("minimum_participants", 1)),
            normalization=str(wbft_cfg.get("normalization", "auto")),
        )
        return consensus.decide(responses), responses

    def _run_ours_consensus(
        self,
        *,
        question: MMLUProQuestion,
        events: list[CrewAIEvent],
        task_text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        assert self.weight_manager is not None
        ours_cfg = self.config.ours
        verification_cfg = dict(ours_cfg.get("verification", {}))
        consensus_cfg = dict(ours_cfg.get("consensus", {}))
        evaluator = AutoGenProposalEvaluator(
            model=self.config.default_model,
            temperature=self.config.temperature,
            verifier_models={
                agent_id: spec.model for agent_id, spec in self._agent_specs.items()
            },
            dimension_weights=dict(verification_cfg.get("dimension_weights", {})),
            fallback_evaluator=HeuristicProposalEvaluator(
                dimension_weights=dict(verification_cfg.get("dimension_weights", {})) or None
            ),
        )
        engine = VerificationEngine(evaluator=evaluator)
        max_per_agent = int(ours_cfg.get("max_memory_proposals_per_agent_per_case", 1))
        include_proposer = bool(verification_cfg.get("include_proposer_as_verifier", False))
        self_threshold = float(verification_cfg.get("self_confidence_threshold", 0.6))
        decisions: list[dict[str, Any]] = []
        proposals: list[dict[str, Any]] = []
        accepted_proposals = []
        counts_by_agent: dict[str, int] = {}
        for event in events:
            source = event.source
            if source not in self._agent_specs:
                continue
            if counts_by_agent.get(source, 0) >= max_per_agent:
                continue
            payload = _extract_memory_proposal_payload(event.content)
            if payload is None:
                continue
            counts_by_agent[source] = counts_by_agent.get(source, 0) + 1
            proposal = self.proposal_builder.from_agent_output(
                task_id=question.question_id,
                agent_id=source,
                output=payload,
                parent_proposals=[item.proposal_id for item in accepted_proposals],
            )
            context = VerificationContext(
                task_id=question.question_id,
                task_description=task_text,
                related_proposals=accepted_proposals,
            )
            self_vector = engine.evaluate(proposal, context, verifier_agent_id=source)
            proposal = _proposal_with_self_verification(proposal, self_vector)
            if self_vector.confidence_score < self_threshold:
                proposals.append(_proposal_payload(proposal, lifecycle="self_rejected"))
                continue
            verifications = []
            if include_proposer:
                verifications.append(self_vector)
            for agent_id in self._agent_specs:
                if agent_id == source:
                    continue
                verifications.append(engine.evaluate(proposal, context, verifier_agent_id=agent_id))
            consensus = self._build_ours_consensus(consensus_cfg)
            decision = consensus.decide(proposal, verifications)
            proposal = _proposal_with_consensus_result(proposal, decision)
            self._update_weights(source, decision)
            decision_payload = decision.to_dict()
            decisions.append(decision_payload)
            lifecycle = "consensus_accepted" if decision.accepted else "consensus_rejected"
            proposals.append(_proposal_payload(proposal, lifecycle=lifecycle, decision=decision_payload))
            if decision.accepted:
                accepted_proposals.append(proposal)
        return decisions, proposals

    def _build_ours_consensus(self, consensus_cfg: dict[str, Any]) -> Any:
        assert self.weight_manager is not None
        if consensus_cfg.get("strategy", "smart_quorum") == "majority_vote":
            return MajorityVoteConsensus(
                confidence_threshold=float(consensus_cfg.get("confidence_threshold", 0.6)),
                majority_threshold=float(consensus_cfg.get("majority_threshold", 0.5)),
                strict_majority=bool(consensus_cfg.get("strict_majority", True)),
            )
        agent_weights = {
            agent_id: self.weight_manager.weight(agent_id)
            for agent_id in self._agent_specs
        }
        honest_agents = [
            agent_id for agent_id, spec in self._agent_specs.items() if not spec.is_byzantine
        ]
        byzantine_agents = [
            agent_id for agent_id, spec in self._agent_specs.items() if spec.is_byzantine
        ]
        return SmartQuorumConsensus(
            agent_weights=agent_weights,
            honest_agents=honest_agents,
            byzantine_agents=byzantine_agents,
            confidence_threshold=float(consensus_cfg.get("confidence_threshold", 0.6)),
            majority_threshold=float(consensus_cfg.get("majority_threshold", 0.5)),
            strict_majority=bool(consensus_cfg.get("strict_majority", True)),
            epsilon_ratio=float(consensus_cfg.get("epsilon_ratio", 0.1)),
            use_dynamic_estimate=bool(consensus_cfg.get("use_dynamic_estimate", False)),
        )

    def _build_weight_manager(self) -> WeightManager:
        cfg = dict(self.config.ours.get("weight_manager", {}))
        return WeightManager(
            alpha=float(cfg.get("alpha", 0.6)),
            beta=float(cfg.get("beta", 0.4)),
            gamma=float(cfg.get("gamma", 5.0)),
            theta=float(cfg.get("theta", 0.5)),
            proposal_window=int(cfg.get("proposal_window", 20)),
            vote_window=int(cfg.get("vote_window", 30)),
            initial_vc=float(cfg.get("initial_vc", 0.5)),
            initial_hc=float(cfg.get("initial_hc", 0.5)),
        )

    def _ensure_memory_backend(self) -> Mem0MemoryBackend:
        if self.memory_backend is None:
            self.memory_backend = Mem0MemoryBackend(
                topk=self.config.memory_topk,
                default_user_id=self.memory_user_id or "exp1_mmlu_crewai_shared",
            )
        return self.memory_backend

    def _upload_accepted_memory_proposals(
        self,
        proposals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        uploads: list[dict[str, Any]] = []
        accepted = [
            proposal
            for proposal in proposals
            if proposal.get("_experiment", {}).get("lifecycle") == "consensus_accepted"
        ]
        if not accepted:
            return uploads
        backend = self._ensure_memory_backend()
        for proposal in accepted:
            try:
                response = backend.add(
                    json.dumps(proposal, ensure_ascii=False),
                    user_id=self.memory_user_id,
                    metadata={
                        "source": "exp1_mmlu_crewai",
                        "proposal_id": proposal.get("header", {}).get("proposal_id", ""),
                        "task_id": proposal.get("header", {}).get("task_id", ""),
                        "agent_id": proposal.get("header", {}).get("agent_id", ""),
                        "consensus_result": proposal.get("verification", {})
                        .get("consensus_result", {})
                        .get("result", ""),
                    },
                )
                uploads.append(
                    {
                        "proposal_id": proposal.get("header", {}).get("proposal_id", ""),
                        "user_id": self.memory_user_id,
                        "uploaded": True,
                        "response": response,
                    }
                )
            except Exception as exc:
                uploads.append(
                    {
                        "proposal_id": proposal.get("header", {}).get("proposal_id", ""),
                        "user_id": self.memory_user_id,
                        "uploaded": False,
                        "error": str(exc),
                    }
                )
        return uploads

    def _update_weights(self, proposer: str, decision: Any) -> None:
        assert self.weight_manager is not None
        confidence = float(decision.metadata.get("proposal_confidence_score", 0.0))
        self.weight_manager.record_proposal_confidence(proposer, confidence)
        self.weight_manager.record_vote_alignment(proposer, decision.accepted)
        for vote in decision.votes:
            self.weight_manager.record_vote_alignment(
                vote.voter_agent_id,
                vote.accept == decision.accepted,
            )
        decision.metadata["weight_snapshots_after_update"] = {
            agent_id: self.weight_manager.snapshot(agent_id)
            for agent_id in sorted(self._agent_specs)
        }

    def _weight_snapshots(self) -> dict[str, Any]:
        if self.weight_manager is None:
            return {}
        return {
            agent_id: self.weight_manager.snapshot(agent_id)
            for agent_id in sorted(self._agent_specs)
        }

    def _model_for_agent(self, is_byzantine: bool, regime: str) -> str:
        if regime == "weak_byzantine_strong_honest":
            return self.config.weak_model if is_byzantine else self.config.strong_model
        if regime == "strong_byzantine_weak_honest":
            return self.config.strong_model if is_byzantine else self.config.weak_model
        return self.config.default_model


def build_crewai_group_matrix(config: Exp1Config) -> list[Exp1GroupSpec]:
    groups: list[Exp1GroupSpec] = []
    for group in config.group_specs:
        if group.method in CREWAI_METHODS:
            groups.append(group)
    return groups


def _prepare_crewai_runtime() -> None:
    root = Path(".crewai_runtime").resolve()
    storage = root / "storage"
    appdata = root / "appdata"
    storage.mkdir(parents=True, exist_ok=True)
    appdata.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOCALAPPDATA", str(appdata))
    os.environ["LOCALAPPDATA"] = str(appdata)
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    import appdirs

    appdirs.user_data_dir = lambda appname=None, appauthor=None, **kwargs: str(storage)


def _build_crewai_llm(model: str, temperature: float | None) -> Any:
    _prepare_crewai_runtime()
    from crewai import LLM

    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key:
        raise ValueError("No model API key configured. Set API_KEY in .env or environment.")
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "provider": "openai",
        "temperature": temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return LLM(**kwargs)


def _task_output_text(task_output: Any) -> str:
    raw = getattr(task_output, "raw", None)
    if raw:
        return str(raw)
    return str(task_output)


def _trace(events: list[CrewAIEvent]) -> str:
    return "\n\n".join(f"{event.source}: {event.content}" for event in events if event.content)


def _token_usage(
    *,
    events: list[CrewAIEvent],
    trace: str,
    proposals: list[dict[str, Any]],
    usage: Any | None,
) -> dict[str, Any]:
    prompt_tokens = _usage_value(usage, "prompt_tokens", "input_tokens", "total_prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens", "output_tokens", "total_completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens", "total_token_count")
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
    if total_tokens is None:
        total_tokens = _estimate_tokens(trace)
        source = "estimated"
    else:
        source = "api_usage"
    proposal_tokens = sum(_estimate_tokens(json.dumps(proposal, ensure_ascii=False)) for proposal in proposals)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_source": source,
        "memory_proposal_tokens": proposal_tokens,
        "memory_proposal_token_source": "estimated",
    }


def _usage_value(usage: Any | None, *names: str) -> int | None:
    if usage is None:
        return None
    for name in names:
        if isinstance(usage, dict) and usage.get(name) is not None:
            return int(usage[name])
        if getattr(usage, name, None) is not None:
            return int(getattr(usage, name))
    return None


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def _extract_memory_proposal_payload(content: str) -> dict[str, Any] | None:
    marker = "MEMORY_PROPOSAL"
    if marker not in content:
        return None
    block = content.split(marker, 1)[1]
    if "END_MEMORY_PROPOSAL" in block:
        block = block.split("END_MEMORY_PROPOSAL", 1)[0]
    import re

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", block, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else block[block.find("{") : block.rfind("}") + 1]
    if not candidate or not candidate.startswith("{"):
        return None
    payload = _loads_json_object(candidate)
    if payload is None:
        return None
    return payload if isinstance(payload, dict) else None


def _loads_json_object(candidate: str) -> dict[str, Any] | None:
    import re

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        repaired = repaired.replace('""data"', '"data"')
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _proposal_with_self_verification(proposal: Any, vector: VerificationVector) -> Any:
    from dataclasses import replace

    return replace(
        proposal,
        verification=replace(
            proposal.verification,
            self_verification=SelfVerificationScores(
                veracity_score=vector.veracity,
                rationality_score=vector.rationality,
                value_score=vector.value,
                security_score=vector.security,
            ),
        ),
    )


def _proposal_with_consensus_result(proposal: Any, decision: Any) -> Any:
    from dataclasses import replace

    payload = decision.metadata.get("consensus_result", {})
    return replace(
        proposal,
        verification=replace(
            proposal.verification,
            multi_verification=MultiVerificationSummary(
                weighted_scores=dict(decision.metadata.get("multi_verification_summary", {}))
            ),
            consensus_result=ConsensusResult(
                total_weight=float(payload.get("total_weight", decision.total_weight)),
                vote_weight=float(payload.get("vote_weight", decision.accept_weight)),
                result=str(payload.get("result", decision.result)),
            ),
        ),
    )


def _proposal_payload(
    proposal: Any,
    *,
    lifecycle: str,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = proposal.to_dict()
    data["_experiment"] = {
        "lifecycle": lifecycle,
        "consensus_result": decision.get("result") if decision else None,
        "token_count": _estimate_tokens(json.dumps(data, ensure_ascii=False)),
        "token_source": "estimated",
    }
    return data


__all__ = [
    "CREWAI_METHODS",
    "CrewAIMMLURunner",
    "build_crewai_group_matrix",
    "load_mmlu_pro_questions",
    "summarize_group_results",
]
