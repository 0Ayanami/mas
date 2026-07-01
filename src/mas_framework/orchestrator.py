from __future__ import annotations

from pathlib import Path
from mas_framework.agents import AgentProtocol, create_agent
from mas_framework.consensus import EventDrivenHotStuffPolicy, SmartQuorumPolicy
from mas_framework.models import AgentConfig, ConsensusResult, MemoryProposal, ProposalStatus
from mas_framework.utils.loader import load_system_prompts
from mas_framework.proposal_tools import Proposal_Tools
import uuid

# Load system prompt once as a string
_SYSTEM_PROMPT = load_system_prompts()

DEFAULT_AGENTS = [
    AgentConfig(
        agent_id="researcher_1",
        role="Researcher",
        system_prompt=_SYSTEM_PROMPT,
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
        self.agent_count = len(self.agents)
        self.policy = policy or EventDrivenHotStuffPolicy()

    def propose_memory_from_trace(
        self,
        *,
        agent_id: str,
        task_id: str,
        react_trace: str,
        task_context: str = "",
        parent_proposal_ids: list[str] | None = None,
    ) -> MemoryProposal | None:
        
        agent = self.agents.get(agent_id)
        if agent is None:
            raise KeyError(f"Unknown or unavailable agent: {agent_id}")

        flag = Proposal_Tools.should_propose_memory(
            agent,
            react_trace=react_trace,
            task_context=task_context,
        )
        if not flag.get("propose_memory"):
            return None

        proposal = Proposal_Tools.create_proposal(
            agent=agent,
            task_id=task_id,
            proposal_id=str(uuid.uuid4()),
            react_trace=react_trace,
            memory_type=str(flag.get("memory_type", "research_note")),
            parent_proposal_ids=parent_proposal_ids,
            task_context=task_context,
        )

        if not Proposal_Tools.self_verify(agent, proposal):
            # 自我验证阶段未通过
            proposal.consensus_result = ConsensusResult(
            voting_agents=0,
            total_agents=self.agent_count,
            vote_weight=0.0,
            total_weight=0.0,
            result=proposal.status,
        )
            return proposal
        
        return self.verify_and_commit(proposal)

    def verify_and_commit(self, proposal: MemoryProposal) -> MemoryProposal:
        proposer = proposal.agent_id

        # 在memory被propose之前 agent在本地已经进行了一次验证,收集系统中其他的agent作为验证器
        validators = [
            agent
            for agent_id, agent in self.agents.items()
            if agent_id != proposer and agent is not None
        ]
        
        # 这里proposal.verifications中应该已经有一个agent的自我验证的verification_vector
        for agent in validators:
            v = Proposal_Tools.verify(agent, proposal) # 每个agent验证proposal之后返回一个verification_vector
            proposal.verifications.append(v)
        
        # 进行共识决策
        proposal.consensus_round += 1
        decision = self.policy.decide(proposal, agent_count=self.agent_count)
        proposal.consensus_result = decision
        proposal.status = decision.result

        if proposal.status == ProposalStatus.ACCEPTED:
            for agent_id, agent in self.agents.items():
                if agent is None:
                    continue
                try:
                    agent.memory.add_proposal(proposal, user_id=agent_id)
                except Exception as exc:
                    print(f"Failed to update memory for proposal {proposal.header.proposal_id}: {exc}")
        
        Proposal_Tools.update_state(agent=self.agents[proposer], 
                                    avg_confidence=proposal.multi_confidence,
                                    status=proposal.status)
        return proposal
