from __future__ import annotations

import os
import json
import re
from typing import Protocol
import math
from mas_framework.memory import Mem0MemoryBackend
from mas_framework.models import AgentConfig, MemoryProposal, VerificationVector, AgentState, SelfVerification, ProposalStatus, MultiAgentVerificationSummary
from mas_framework.tools import build_default_tool_registry, ToolRegistry
from mas_framework.utils.loader import load_verify_prompts, load_memory_proposal_skill
from jinja2 import Template


class AgentProtocol(Protocol):
    config: AgentConfig

    def run(self, prompt: str) -> str:
        """
        Run the agent and return its response.
        """
        ...


class Agent:
    def __init__(self, config: AgentConfig, memory: Mem0MemoryBackend | None = None, tools: ToolRegistry | None = None):
        self.config = config
        self.memory = memory or Mem0MemoryBackend()
        self.tools = tools or build_default_tool_registry([self.memory.search]).get_tools()
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
            tools=self.tools,
        )

    def run(self, prompt: str) -> str:
        from camel.messages import BaseMessage

        message = BaseMessage.make_user_message(role_name=self.config.role, content=prompt)
        response = self._agent.step(message)
        if isinstance(response, tuple):
            response = response[0]
        return response.msgs[0].content


def create_agent(config: AgentConfig) -> AgentProtocol | None:
    """Factory function to create an agent with CAMEL backend."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print(f"Warning: No API key found for {config.agent_id}. Agent will not be backed by LLM.")
        return None
    try:
        return Agent(config)
    except Exception as exc:
        print(f"Agent init failed for {config.agent_id}; Error: {exc}")
        return None
