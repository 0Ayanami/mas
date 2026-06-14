from __future__ import annotations

from pathlib import Path
from mas_framework.agents import AgentProtocol, create_agent
from mas_framework.consensus import SmartQuorumPolicy
from mas_framework.memory import Mem0MemoryBackend
from mas_framework.models import AgentConfig, ConsensusDecision, MemoryProposal, ProposalStatus
from mas_framework.utils.loader import load_system_prompts
from mas_framework.proposal_tools import Proposal_Tools

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
        self.policy = policy or SmartQuorumPolicy()

    def verify_and_commit(self, proposal: MemoryProposal) -> ConsensusDecision:
        proposer = proposal.header.proposing_agent_id

        # 在memory被propose之前 agent在本地已经进行了一次验证
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
        decision = self.policy.decide(proposal, agent_count=len(self.agents))
        proposal.status = decision.status
        if proposal.status == ProposalStatus.ACCEPTED:
            for agent_id, agent in self.agents.items():
                try:
                    agent.memory.add_proposal(proposal, user_id=agent_id)
                except Exception as exc:
                    print(f"Failed to update memory for proposal {proposal.header.proposal_id}: {exc}")
        
        Proposal_Tools.update_state(agent=self.agents[proposer], 
                                    mac=proposal.verification.multi_agent_verification,
                                    status=decision.status)
        return decision
