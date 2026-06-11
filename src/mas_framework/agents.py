from __future__ import annotations

import os
import json
import re
from typing import Protocol
import math
from mas_framework.memory import Mem0MemoryBackend
from mas_framework.models import AgentConfig, MemoryProposal, VerificationVector,AgentState, SelfVerification, ProposalStatus, MultiAgentVerificationSummary
from mas_framework.tools import build_default_tool_registry, create_proposal_creation_toolkit, ToolRegistry


class AgentProtocol(Protocol):
    config: AgentConfig

    def run(self, prompt: str) -> str:
        """
        Run the agent and return its response.
        """
        ...

    def verify(self, proposal: MemoryProposal) -> VerificationVector:
        """
        Verify a memory proposal and return a verification vector.
        """
        ...

    def propose_memory(self, proposal_data: MemoryProposal) -> MemoryProposal:
        """
        Build a memory proposal with self-verification applied.
        """
        ...

class Agent:
    def __init__(self, config: AgentConfig, memory: Mem0MemoryBackend | None = None, tools: ToolRegistry | None = None):
        self.config = config
        self.memory = memory or Mem0MemoryBackend()
        self.tools = tools or build_default_tool_registry(memory_search=self.memory.search)
        self._agent = self._build_agent()
        self.state = AgentState()

    def _build_agent(self):
        from camel.agents import ChatAgent
        from camel.messages import BaseMessage
        from camel.models import ModelFactory

        model = ModelFactory.create(
            model_platform=self.config.model_platform,
            model_type=self.config.model_type,
            model_config_dict=self.config.model_config_dict,
        )

        system_message = BaseMessage.make_assistant_message(
            role_name=self.config.role,
            content=self.config.system_prompt,
        )
        return ChatAgent(
            model=model,
            system_message=system_message,
            tools=self.tools.camel_tools(),
        )

    def run(self, prompt: str) -> str:
        from camel.messages import BaseMessage

        message = BaseMessage.make_user_message(role_name=self.config.role, content=prompt)
        response = self._agent.step(message)
        return response.msgs[0].content

    def _compute_agent_weight(self, alpha: float = 0.5, beta: float = 0.5) -> float:
        """
        根据agent的历史表现计算其权重
        """
        vc = self.state.get("verified_conf", 0.0)
        hc = self.state.get("historical_conf", 0.0)
        base = self.state.get("base", 1.0)
        q = alpha * vc + beta * hc
        return float(math.exp(base * q))
    
    def verify(self, proposal: MemoryProposal) -> VerificationVector:
        prompt = f"""
Evaluate this memory proposal for Byzantine-resilient MAS research.-
Return a JSON object ONLY (no extra text) with these keys:
{{
  "veracity": true or false,
  "rationality": true or false,
  "value": true or false,
  "security": true or false,
  "rationale": "string explaining the judgement"
}}

Proposal:
{proposal.model_dump_json(indent=2)}
"""
        response = self.run(prompt)

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find JSON in verifier response: {response}")

        payload_str = match.group(0)
        try:
            payload = json.loads(payload_str)
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON from verifier response: {exc}\nResponse: {response}")

        def to_bool(value: object) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                v = value.strip().lower()
                return v in ("true", "t", "yes", "y", "1")
            return False

        veracity = to_bool(payload.get("veracity"))
        rationality = to_bool(payload.get("rationality"))
        value_flag = to_bool(payload.get("value"))
        security = to_bool(payload.get("security"))
        rationale = str(payload.get("rationale", "")).strip()

        return VerificationVector.from_binary_votes(
            veracity=veracity,
            rationality=rationality,
            value=value_flag,
            security=security,
            rationale=rationale,
            conf_threshold=self.config.conf_threshold,
            verifier_id=self.config.agent_id,
            weight=self.state["weight"],
        )

    def self_verify(self, proposal: MemoryProposal):
        vector = self.verify(proposal)
        if vector.vote_result:
            proposal.verifications = []
            proposal.verifications.append(vector)
            sf = SelfVerification(veracity=vector.veracity, rationality=vector.rationality, 
                                  value=vector.value, security=vector.security, confidence=vector.confidence,
                                    rationale=vector.rationale)
            proposal.verification.self_verification = sf
            self.submit_proposal(proposal)
        return None
    
    def submit_proposal(self, proposal: MemoryProposal):    
        pass

    def update_state(self, mac: MultiAgentVerificationSummary, status:ProposalStatus):
        # 更新agent state
        self.state["proposal_sum"] += 1
        if status == ProposalStatus.ACCEPTED:
            self.state["proposal_submitted"] += 1
        self.state["historical_conf"] = round(self.state["proposal_submitted"] / self.state["proposal_sum"], 4) if self.state["proposal_sum"] > 0 else 0.0

        if mac and mac.confidence is not None:
            self.state["verified_conf_sum"] += mac.confidence
            self.state["verified_conf"] = round(self.state["verified_conf_sum"] / self.state["proposal_sum"], 4) if self.state["proposal_sum"] > 0 else 0.0
        self.state["weight"] = self._compute_agent_weight()

def create_agent(config: AgentConfig) -> AgentProtocol:
    if bool(os.getenv("OPENAI_API_KEY")):
        try:
            return Agent(config)
        except Exception as exc:
            print(f"Agent init failed for {config.agent_id}; Error: {exc}")
    return None
