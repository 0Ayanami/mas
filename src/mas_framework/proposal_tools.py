from mas_framework.agents import Agent
import math
from mas_framework.prompts import load_verify_prompts
from mas_framework.models import (VerificationVector, MemoryProposal, ProposalStatus, 
                                  SelfVerification, MultiAgentVerificationSummary, AgentState)
from jinja2 import Template
import re
import json

class Proposal_Tools:
    def __init__(self):
        pass
    
    @staticmethod
    def _compute_agent_weight(state: AgentState, alpha: float = 0.5, beta: float = 0.5) -> float:
        """
        根据agent的历史表现计算其权重
        """
        vc = state.get("verified_conf", 0.0)
        hc = state.get("historical_conf", 0.0)
        base = state.get("base", 1.0)
        q = alpha * vc + beta * hc
        return float(math.exp(base * q))

    @staticmethod
    def create_proposal():
        """
    Header：proposal id（需要在mas全局中unique）, task id（orchestrator在工作流中分配）, timestamp(有默认值), proposing agent signature(使用agent id), 
            parent proposal list（可为空） ,message body hash(), proposal summary（需要agent.step）
    
    Body（以下字段按实际情况填写，部分内容可以留空）:Thoughts：thoughts abstract(思考路径关键信息摘要), key decision points & decision results(涉及到的主要决策点信息和决策结果摘要)
        Action：action list(执行的操作列表，如action_1: api function call...; action_2: web sesearch with keywords...; action_3: interaction with agent...)
        Data：data list(任务相关的关键信息/数据列表，agent本地检索到的提供关键信息摘要，公开渠道获取的提供关键信息摘要和访问链接等)
        Observations：result list(当前取得的主要结果或观测情况列表，如result_1: complete subtask_i...; result_2: fetched data from url...)
        """
        pass
    
    @classmethod
    def verify(cls, agent: Agent, proposal: MemoryProposal) -> VerificationVector:
        template = Template(load_verify_prompts())
        prompt = template.render(proposal=proposal.model_dump_json(indent=2))
        
        response = agent.run(prompt)

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
            conf_threshold=agent.config.conf_threshold,
            verifier_id=agent.config.agent_id,
            weight=agent.state.get("weight", 0.0),
        )
    
    @classmethod
    def self_verify(cls, agent: Agent, proposal: MemoryProposal) -> bool:
        """Self-verify and submit the proposal if it passes."""
        vector = cls.verify(agent, proposal)
        if vector.vote_result:
            proposal.verifications = []
            proposal.verifications.append(vector)
            proposal.verification.self_verification = SelfVerification(
                veracity=vector.veracity,
                rationality=vector.rationality,
                value=vector.value,
                security=vector.security,
                confidence=vector.confidence,
                rationale=vector.rationale,
            )
        else:
            proposal.status = ProposalStatus.REJECTED
            agent.state.proposal_sum += 1
        return vector.vote_result
    
    @classmethod
    def update_state(cls, agent: Agent, mac: MultiAgentVerificationSummary, status: ProposalStatus):
        """Update agent state after consensus decision."""
        agent.state.proposal_sum += 1
        if status == ProposalStatus.ACCEPTED:
            agent.state.proposal_submitted += 1
        agent.state.historical_conf = round(
            agent.state.proposal_submitted / agent.state.proposal_sum, 4
        ) if agent.state.proposal_sum > 0 else 0.0

        if mac and mac.confidence is not None:
            agent.state.verified_conf_sum += mac.confidence
            agent.state.verified_conf = round(
                agent.state.verified_conf_sum / agent.state.proposal_sum, 4
            ) if agent.state.proposal_sum > 0 else 0.0
        agent.state.weight = cls._compute_agent_weight(agent.state)
    
    @staticmethod
    def submit_proposal(agent: Agent, proposal: MemoryProposal):
        """Submit the proposal to the orchestrator for multi-agent verification."""
        pass