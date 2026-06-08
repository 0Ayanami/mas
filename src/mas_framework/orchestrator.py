from __future__ import annotations

from pathlib import Path

from mas_framework.agents import AgentProtocol, create_agent
from mas_framework.consensus import SmartQuorumPolicy
from mas_framework.memory import SQLiteMemoryStore
from mas_framework.models import AgentConfig, ConsensusDecision, MemoryProposal, ProposalStatus
from mas_framework.tools import ToolRegistry, build_default_tool_registry


DEFAULT_AGENTS = [
    AgentConfig(
        agent_id="researcher_1",
        role="Researcher",
        system_prompt=(
            "You collect and summarize evidence for consensus-based multi-agent memory research. "
            "Prefer concrete claims, assumptions, and open questions."
        ),
    ),
    AgentConfig(
        agent_id="method_critic",
        role="MethodCritic",
        system_prompt=(
            "You inspect proposals for methodological rigor, missing assumptions, and Byzantine risks."
        ),
    ),
    AgentConfig(
        agent_id="security_verifier",
        role="SecurityVerifier",
        system_prompt=(
            "You verify memory proposals for factuality, rationality, usefulness, and malicious content."
        ),
    ),
    AgentConfig(
        agent_id="systems_verifier",
        role="SystemsVerifier",
        system_prompt=(
            "You evaluate distributed-systems feasibility, quorum implications, and protocol fit."
        ),
    ),
]


class ResearchOrchestrator:
    def __init__(
        self,
        *,
        memory: SQLiteMemoryStore | None = None,
        tools: ToolRegistry | None = None,
        agent_configs: list[AgentConfig] | None = None,
        policy: SmartQuorumPolicy | None = None,
    ):
        self.memory = memory or SQLiteMemoryStore()
        self.tools = tools or build_default_tool_registry(memory_search=self._search_memory_tool)
        self.agent_configs = agent_configs or DEFAULT_AGENTS
        self.agents: dict[str, AgentProtocol] = {
            config.agent_id: create_agent(config, self.tools) for config in self.agent_configs
        }
        self.policy = policy or SmartQuorumPolicy()
        # agent_stats holds running tallies used to compute voting weights
        # fields: proposals_submitted, proposals_passed, confidence_sum, confidence_count
        self.agent_stats: dict[str, dict[str, float]] = {
            config.agent_id: {
                "proposals_submitted": 0,
                "proposals_passed": 0,
                "confidence_sum": 0.0,
                "confidence_count": 0,
            }
            for config in self.agent_configs
        }

    def _compute_agent_weight(self, agent_id: str, *, alpha: float = 0.5, beta: float = 0.5, lambda_coeff: float = 1.0) -> float:
        stats = self.agent_stats.get(agent_id, None)
        # default neutral values if no history
        if not stats:
            vc = 0.5
            avg_conf = 0.5
        else:
            submitted = stats.get("proposals_submitted", 0)
            passed = stats.get("proposals_passed", 0)
            vc = (passed / submitted) if submitted > 0 else 0.5
            conf_count = stats.get("confidence_count", 0)
            avg_conf = (stats.get("confidence_sum", 0.0) / conf_count) if conf_count > 0 else 0.5
        q = alpha * vc + beta * avg_conf
        import math

        return float(math.exp(lambda_coeff * q))

    def _search_memory_tool(self, query: str, limit: int = 5) -> str:
        proposals = self.memory.search(query=query, limit=limit)
        if not proposals:
            return "No memory records matched the query."
        return "\n".join(
            f"- {proposal.status.value}: {proposal.short_label()} :: {proposal.results_observation}"
            for proposal in proposals
        )

    def run_document_research(
        self,
        *,
        document_path: str,
        task_id: str = "consensus-memory-research",
    ) -> tuple[MemoryProposal, ConsensusDecision]:
        document = self.tools.call("read_text_file", path=document_path, max_chars=18000)
        focused_lines = self.tools.call(
            "keyword_extract",
            text=document,
            keywords="Consensus, Memory Proposal, Verification, Quorum, Evaluation, 共识, 记忆",
        )

        researcher = self.agents["researcher_1"]
        research_prompt = f"""
Read the research notes and produce a compact first-step implementation proposal for a multi-agent system with tool calling and memory.

Document path: {Path(document_path).as_posix()}

Focused notes:
{focused_lines}
"""
        analysis = researcher.run(research_prompt)
        proposal = MemoryProposal(
            agent_id=researcher.config.agent_id,
            task_id=task_id,
            memory_type="research_note",
            title="Initial CAMEL-based MAS scaffold",
            thoughts_decision=(
                "The first implementation step should create minimal but explicit abstractions "
                "for agents, tools, shared memory, proposal schemas, verification, and quorum decisions."
            ),
            action="Read research markdown, extracted consensus/memory requirements, and drafted scaffold.",
        data={
                "document_path": str(document_path),
                "focused_lines": focused_lines,
                "agent_analysis": analysis,
                "tool_calls": [tool_call.model_dump() for tool_call in self.tools.history],
            },
            results_observation=(
                "A basic research workflow can produce structured MemoryProposal records, "
                "ask multiple verifier agents to evaluate them, and persist accepted/rejected "
                "records for later consensus-protocol experiments."
            ),
            self_confidence=0.82,
        )
        decision = self.verify_and_commit(proposal)
        return proposal, decision

    def verify_and_commit(self, proposal: MemoryProposal) -> ConsensusDecision:
        validators = [
            agent
            for agent_id, agent in self.agents.items()
            if agent_id != proposal.agent_id
        ]
        # collect verification vectors and attach per-verifier weights based on history
        verifications = []
        for agent in validators:
            v = agent.verify(proposal)
            # compute weight for this verifier (based on its stats)
            weight = self._compute_agent_weight(agent.config.agent_id)
            # set weight on the verification vector
            try:
                setattr(v, "weight", weight)
            except Exception:
                pass
            verifications.append(v)
        proposal.verifications = verifications
        decision = self.policy.decide(proposal, validator_count=len(validators))
        proposal.status = decision.status
        self.memory.save_proposal(proposal)
        # update proposing agent stats
        proposer = proposal.agent_id
        stats = self.agent_stats.setdefault(proposer, {
            "proposals_submitted": 0,
            "proposals_passed": 0,
            "confidence_sum": 0.0,
            "confidence_count": 0,
        })
        stats["proposals_submitted"] = stats.get("proposals_submitted", 0) + 1
        if decision.status == ProposalStatus.ACCEPTED:
            stats["proposals_passed"] = stats.get("proposals_passed", 0) + 1
        # record the proposal's overall multi-agent confidence (if available)
        mac = proposal.verification.multi_agent_verification
        if mac and mac.confidence is not None:
            stats["confidence_sum"] = stats.get("confidence_sum", 0.0) + float(mac.confidence)
            stats["confidence_count"] = stats.get("confidence_count", 0) + 1
        return decision
