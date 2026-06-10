from __future__ import annotations

import os
import json
import re
from typing import Protocol

from mas_framework.memory import Mem0MemoryBackend
from mas_framework.models import AgentConfig, MemoryProposal, VerificationVector
from mas_framework.tools import build_default_tool_registry


class AgentProtocol(Protocol):
    config: AgentConfig

    def run(self, prompt: str) -> str:
        """
        运行智能体，返回结果。
        """
        ...

    def verify(self, proposal: MemoryProposal) -> VerificationVector:
        """
        验证memory proposal，返回验证向量。
        """
        ...

class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.memory = config.memory or Mem0MemoryBackend()
        self.tools = config.tools or build_default_tool_registry(memory_search=config.memory.search if config.memory else None)
        self._agent = self._build_agent()

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
        return ChatAgent(model=model, system_message=system_message, tools=self.tools.camel_tools())

    def run(self, prompt: str) -> str:
        from camel.messages import BaseMessage

        message = BaseMessage.make_user_message(role_name="ResearchCoordinator", content=prompt)
        response = self._agent.step(message)
        return response.msgs[0].content

    def verify(self, proposal: MemoryProposal) -> VerificationVector:
        prompt = f"""
Evaluate this memory proposal for Byzantine-resilient MAS research.
Return a JSON object ONLY (no extra text) with these keys:
{
  "veracity": true or false,
  "rationality": true or false,
  "value": true or false,
  "security": true or false,
  "rationale": "string explaining the judgement"
}

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
            verifier_id=self.config.agent_id,
        )


def create_agent(config: AgentConfig) -> AgentProtocol:
    if bool(os.getenv("OPENAI_API_KEY")):
        try:
            return Agent(config)
        except Exception as exc:
            print(f"Agent init failed for {config.agent_id}; Error: {exc}")
    return None
