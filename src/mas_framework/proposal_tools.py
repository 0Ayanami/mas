from typing import Any

from mas_framework.agents import Agent
import math
from mas_framework.utils.loader import (load_propose_prompts, load_memory_proposal_skill,
                                         load_verify_prompts, load_create_proposal_prompts)
from mas_framework.models import (VerificationVector, MemoryProposal, ProposalStatus, 
                                  SelfVerification, MultiAgentVerificationSummary, AgentState,
                                  ProposalHeader, ProposalBody)
from jinja2 import Template
import re
import json
import hashlib

class Proposal_Tools:
    def __init__(self):
        pass
    
    @staticmethod
    def _compute_agent_weight(state: AgentState, alpha: float = 0.5, beta: float = 0.5) -> float:
        """
        根据agent的历史表现计算其权重
        """
        vc = getattr(state, "verified_conf", 0.0)
        hc = getattr(state, "historical_conf", 0.0)
        base = getattr(state, "base", 1.0)
        q = alpha * vc + beta * hc
        return float(math.exp(base * q))

    @staticmethod
    def _extract_json(response: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find JSON in response: {response}")
        try:
            return json.loads(match.group(0))
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON response: {exc}\nResponse: {response}") from exc

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @classmethod
    def should_propose_memory(
        cls,
        agent: Agent,
        *,
        react_trace: str,
        task_context: str = "",
    ) -> dict[str, Any]:
        """Ask the agent whether this completed ReAct cycle should become memory."""
        template = Template(load_propose_prompts())
        prompt = template.render(memory_proposal_skill=load_memory_proposal_skill(),
                                  react_trace=react_trace, task_context=task_context)
        
        res = agent.run(prompt)
        payload = cls._extract_json(res)

        signal = str(payload.get("signal", "")).upper()
        propose = bool(payload.get("propose_memory")) or signal == "MEMORY_PROPOSE"
        payload["propose_memory"] = propose
        payload["signal"] = "MEMORY_PROPOSE" if propose else "NO_MEMORY"

        payload.setdefault("memory_type", "research_note")
        return payload

    @classmethod
    def create_proposal(
        cls,
        *,
        agent: Agent,
        task_id: str,
        proposal_id: str | None = None,
        react_trace: str,
        memory_type: str = "research_note",
        parent_proposal_ids: list[str] | None = None,
        task_context: str = "",
    ) -> MemoryProposal:
        """
        使用prompt让agent总结memory，然后构建memory proposal的header和body。
        """
        template = Template(load_create_proposal_prompts())
        prompt = template.render(react_trace=react_trace, task_context=task_context)

        res = agent.run(prompt)
        payload = cls._extract_json(res)

        summary = str(payload.get("proposal_summary", "")).strip()
        body = ProposalBody(
            thoughts=payload.get("thoughts") or {},
            actions=cls._as_list(payload.get("actions")),
            data=cls._as_list(payload.get("data")),
            observations=cls._as_list(payload.get("observations")),
        )

        # 计算body的hash值
        body_content = json.dumps(
            body.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        body_hash = hashlib.sha256(body_content).hexdigest()

        header = ProposalHeader(
            task_id=task_id,
            proposal_id=proposal_id,
            agent_id=agent.config.agent_id,
            agent_signature=agent.config.agent_id,
            parent_proposals=parent_proposal_ids or [],
            proposal_summary=summary,
            memory_type=memory_type,
            body_hash=body_hash,
        )
        return MemoryProposal(header=header, body=body)
    
    @classmethod
    def verify(cls, agent: Agent, proposal: MemoryProposal) -> VerificationVector:
        template = Template(load_verify_prompts())
        prompt = template.render(proposal=proposal.model_dump_json(indent=2))
        
        response = agent.run(prompt)
        payload = cls._extract_json(response)

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
            weight=getattr(agent.state, "weight", 1.0),
        )
    
    @classmethod
    def self_verify(cls, agent: Agent, proposal: MemoryProposal) -> bool:
        """Self-verify and submit the proposal if it passes."""
        vector = cls.verify(agent, proposal)
        if vector.vote_result:
            proposal.verifications = []
            proposal.verifications.append(vector)
            if proposal.verification is None:
                from mas_framework.models import ProposalVerification

                proposal.verification = ProposalVerification()
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
    def update_state(cls, agent: Agent, avg_confidence: float, status: ProposalStatus):
        """Update agent state after consensus decision."""
        agent.state.proposal_sum += 1
        if status == ProposalStatus.ACCEPTED:
            agent.state.proposal_submitted += 1
        agent.state.historical_conf = round(
            agent.state.proposal_submitted / agent.state.proposal_sum, 4
        ) if agent.state.proposal_sum > 0 else 0.0

        if avg_confidence is not None:
            agent.state.verified_conf_sum += avg_confidence
            agent.state.verified_conf = round(
                agent.state.verified_conf_sum / agent.state.proposal_sum, 4
            ) if agent.state.proposal_sum > 0 else 0.0
        agent.state.weight = cls._compute_agent_weight(agent.state)
