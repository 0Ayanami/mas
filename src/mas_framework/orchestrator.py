from __future__ import annotations

from pathlib import Path
import math
from mas_framework.agents import AgentProtocol, create_agent
from mas_framework.consensus import SmartQuorumPolicy
from mas_framework.memory import Mem0MemoryBackend
from mas_framework.models import AgentConfig, ConsensusDecision, MemoryProposal, ProposalStatus
from mas_framework.tools import ToolRegistry, build_default_tool_registry
 

DEFAULT_AGENTS = [
    AgentConfig(
        agent_id="researcher_1",
        role="Researcher",
        system_prompt=(
            "You collect and summarize evidence for consensus-based multi-agent memory research. "
            "Prefer concrete claims, assumptions, and open questions.\n"
            "\n"
            "Memory Proposal Workflow:\n"
            "After each ReAct step, decide if a Memory Proposal is needed (research_note / "

            "evidence / milestone / tool_observation). If so, call `prepare_proposal_for_submission` "
            "to build the proposal locally (header + body + self-verification), then the "
            "orchestrator handles multi-agent consensus via `verify_and_commit`."
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

class Orchestrator:
    def __init__(
        self,
        *,
        agent_configs: list[AgentConfig] | None = None,
        policy: SmartQuorumPolicy | None = None,
    ):
        self.agent_configs = agent_configs or DEFAULT_AGENTS
        self.agents: dict[str, AgentProtocol] = {
            config.agent_id: create_agent(config) for config in self.agent_configs
        }
        self.policy = policy or SmartQuorumPolicy()

        """ agent state维护的是每个agent提出的memory proposal的统计数据，用于后续计算agent权重
            proposal_sum: agent提出的proposal总数
            proposal_submitted: proposal成功递交总数
            base: 放大系数,由agent的LLM能力决定
            accuracy: proposal成功递交的比例
            verification_quality: 多维度验证综合得分
        """
        self.agent_stats: dict[str, dict[str, float]] = {
            config.agent_id: {
                "proposal_sum": 0,
                "proposal_submitted": 0,
                "base": 1.0,
                "verified_conf": 0.0,
                "verified_conf_sum": 0.0,
                "historical_conf": 0.0,
                "weight": 1.0,
            }
            for config in self.agent_configs
        }
        # 非持久化保存，也就是说Orchestrator重启后agent的历史表现会归零（这一部分可能需要改进）

    def _compute_agent_weight(self, agent_id: str, *, alpha: float = 0.5, beta: float = 0.5) -> float:
        """
        根据agent的历史表现计算其权重
        """
        stats = self.agent_stats.get(agent_id, None)
        if not stats:
            vc = 1.0
            hc = 1.0
            base = 1.0
        else:
            vc = stats.get("verified_conf", 1.0)
            hc = stats.get("historical_conf", 1.0)
            base = stats.get("base", 1.0)
        q = alpha * vc + beta * hc
        return float(math.exp(base * q))

    def verify_and_commit(self, proposal: MemoryProposal) -> ConsensusDecision:
        proposer = proposal.header.proposing_agent_id

        validators = [
            agent
            for agent_id, agent in self.agents.items()
            if agent_id != proposer
        ]
        
        verifications = []
        for agent in validators:
            v = agent.verify(proposal)
            weight = self.agent_stats.get(agent.config.agent_id, {}).get("weight", 1.0)
            try:
                setattr(v, "weight", weight)
            except Exception:
                pass
            verifications.append(v)
        proposal.verifications = verifications

        # 进行共识决策
        proposal.consensus_round += 1
        decision = self.policy.decide(proposal, validator_count=len(validators))
        proposal.status = decision.status
        if proposal.status == ProposalStatus.ACCEPTED:
            for agent_id, agent in self.agents.items():
                try:
                    agent.memory.add_proposal(proposal, user_id=agent_id)
                except Exception as exc:
                    print(f"Failed to update memory for proposal {proposal.ProposalHeader.proposal_id}: {exc}")
        
        # 更新agent state
        stats = self.agent_stats.setdefault(proposer, {
                "proposal_sum": 0,
                "proposal_submitted": 0,
                "base": 1.0,
                "verified_conf": 0.0,
                "verified_conf_sum": 0.0,
                "historical_conf": 0.0,
                "weight": 1.0,
            })
        stats["proposal_sum"] = stats.get("proposal_sum", 0) + 1
        if decision.status == ProposalStatus.ACCEPTED:
            stats["proposal_submitted"] = stats.get("proposal_submitted", 0) + 1
            stats["historical_conf"] = round(stats["proposal_submitted"] / stats["proposal_sum"], 4) if stats["proposal_sum"] > 0 else 0.0

        mac = proposal.verification.multi_agent_verification
        if mac and mac.confidence is not None:
            stats["verified_conf_sum"] = stats.get("verified_conf_sum", 0.0) + mac.confidence
            stats["verified_conf"] = round(stats["verified_conf_sum"] / stats["proposal_sum"], 4) if stats["proposal_sum"] > 0 else 0.0
        stats["weight"] = self._compute_agent_weight(proposer)
        return decision
