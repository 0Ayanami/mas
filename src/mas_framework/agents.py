from __future__ import annotations

import os
import json
import re
from typing import Protocol
import math
from mas_framework.memory import Mem0MemoryBackend
from mas_framework.models import AgentConfig, MemoryProposal, VerificationVector, AgentState, SelfVerification, ProposalStatus, MultiAgentVerificationSummary
from mas_framework.tools import build_default_tool_registry, create_proposal_creation_toolkit, ToolRegistry
from mas_framework.utils.loader import load_verify_prompts
from jinja2 import Template

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
            tools=self.tools.camel_tools(),
        )

    def run(self, prompt: str) -> str:
        from camel.messages import BaseMessage

        message = BaseMessage.make_user_message(role_name=self.config.role, content=prompt)
        response = self._agent.step(message)
        return response.msgs[0].content
    
    def create_proposal(self):
        """
    Header：proposal id（需要在mas全局中unique）, task id（orchestrator在工作流中分配）, timestamp(有默认值), proposing agent signature(使用agent id), 
            parent proposal list（可为空） ,message body hash(), proposal summary（需要agent.step）
    
    Body（以下字段按实际情况填写，部分内容可以留空）:Thoughts：thoughts abstract(思考路径关键信息摘要), key decision points & decision results(涉及到的主要决策点信息和决策结果摘要)
        Action：action list(执行的操作列表，如action_1: api function call...; action_2: web sesearch with keywords...; action_3: interaction with agent...)
        Data：data list(任务相关的关键信息/数据列表，agent本地检索到的提供关键信息摘要，公开渠道获取的提供关键信息摘要和访问链接等)
        Observations：result list(当前取得的主要结果或观测情况列表，如result_1: complete subtask_i...; result_2: fetched data from url...)
        """
        pass

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
        template = Template(load_verify_prompts())
        prompt = template.render(proposal=proposal.model_dump_json(indent=2))
        
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
        """
        agent提出memory proposal，进行本地验证之后提交，开启多智能体验证共识机制。
        """   
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
