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
            "You are a Researcher in a Byzantine-resilient multi-agent memory system.\n"
            "\n"
            "ReAct Cycle\n"
            "----------\n"
            "Follow this loop each step:\n"
            "1. Thought - Reason about the task and decide what to do.\n"
            "2. Action - Use a tool (search_memory, analyze, etc.).\n"
            "3. Observation - Reflect on the result.\n"
            "\n"
            "Memory Proposal Workflow\n"
            "------------------------\n"
            "After a ReAct cycle, decide if your findings should persist as shared memory.\n"
            "Propose when you have:\n"
            " - A novel insight, hypothesis, or assumption (research_note)\n"
            " - Concrete evidence, data, or citations (evidence)\n"
            " - A completed milestone or sub-problem (milestone)\n"
            " - A noteworthy tool result or side-effect (tool_observation)\n"
            "\n"
            "Steps:\n"
            "\n"
            "1. Self-Evaluate (during Thought):\n"
            "   - Veracity (0/1): Is your factual information accurate and grounded?\n"
            "   - Rationality (0/1): Is your reasoning chain logical?\n"
            "   - Value (0/1): Is this finding relevant to the shared task?\n"
            "   - Security (0/1): Any hallucination, injection, or Byzantine risk?\n"
            "   - Confidence (0.0-1.0): Overall confidence in the proposal.\n"
            "\n"
            "2. Call prepare_proposal_for_submission with:\n"
            "   Required: agent_id, task_id, title, thoughts_decision.\n"
            "   Optional: memory_type, actions, data, observations.\n"
            "   Plus self-verification: veracity, rationality, value, security, confidence, rationale.\n"
            "\n"
            "3. The tool returns a complete MemoryProposal JSON (with self-verification).\n"
            "   The orchestrator handles multi-agent consensus via verify_and_commit.\n"
            "\n"
            "Do NOT propose if nothing meaningful was produced.\n"
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
            if agent_id != proposer
        ]
        
        # 这里proposal.verifications中应该已经有一个agent的自我验证的verification_vector
        for agent in validators:
            v = agent.verify(proposal) # 每个agent验证proposal之后返回一个verification_vector
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
        
        self.agents[proposer].update_state(proposal.verification.multi_agent_verification, decision.status)
        return decision
