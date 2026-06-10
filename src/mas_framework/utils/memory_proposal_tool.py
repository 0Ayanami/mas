"""
Memory Proposal Tool for MAS Framework

A comprehensive module for creating, verifying, and submitting memory proposals.
Contains the low-level ProposalBuilder (static methods) and the high-level
MemoryProposalTool (orchestrator-aware), plus tool factory functions.
"""

from __future__ import annotations

import json
from typing import Any

from mas_framework.models import MemoryProposal


# ======================================================================
# Agent Prompt Template - 可集成到 Agent 的 system_prompt 中
# 指导 Agent 在完成 ReAct 循环后何时、如何构建 Memory Proposal
# ======================================================================
REACT_MEMORY_PROPOSAL_PROMPT = """
Memory Proposal Workflow:

After each ReAct (Observation -> Thought -> Action) step, check if any of:
  - research_note: new insight, hypothesis, summary
  - evidence: factual data supporting the task
  - milestone: a sub-goal completed
  - tool_observation: valuable observation from a tool

If yes, call `prepare_proposal_for_submission` with your thoughts, action,
results, and memory_type. The tool builds the full proposal (header + body +
self-verification) locally. Return the JSON - the orchestrator runs
multi-agent consensus via verify_and_commit.
"""


class ProposalBuilder:
    """
    用于帮助Agent构建Memory Proposal的工具类
    包含信息整合、自验证等功能
    """

    @staticmethod
    def create_proposal(
        agent_id: str,
        task_id: str,
        thoughts: str | dict[str, Any],
        actions: str | list[dict[str, Any]],
        observations: str | list[dict[str, Any]],
        data: str | list[dict[str, Any]] | dict[str, Any] | None = None,
        title: str = "New Research Note",
        memory_type: str = "research_note",
        parent_proposal_ids: list[str] | None = None,
        confidence: float = 0.0,
    ) -> MemoryProposal:
        """
        构建一个新的 MemoryProposal 实例。
        """
        # normalize thoughts
        thoughts_decision = json.dumps(
            thoughts if isinstance(thoughts, dict) else {"abstract": thoughts},
            ensure_ascii=False,
        )

        # normalize actions
        if isinstance(actions, str):
            actions_list = [{"action_id": "action_1", "description": actions}]
        else:
            actions_list = list(actions)
        action_str = json.dumps(actions_list, ensure_ascii=False)

        # normalize observations
        if isinstance(observations, str):
            obs_list = [{"observation_id": "obs_1", "description": observations}]
        else:
            obs_list = list(observations)
        obs_str = json.dumps(obs_list, ensure_ascii=False)

        # normalize data
        data_list: list[dict[str, Any]] = []
        if data is not None:
            if isinstance(data, dict):
                data_list = [{"data_id": k, "content": v} for k, v in data.items()]
            elif isinstance(data, list):
                data_list = list(data)
            else:
                data_list = [{"data_id": "data_1", "content": str(data)}]

        return MemoryProposal(
            agent_id=agent_id,
            task_id=task_id,
            memory_type=memory_type,
            title=title,
            thoughts_decision=thoughts_decision,
            action=action_str,
            results_observation=obs_str,
            data=data_list,
            parent_proposal_ids=parent_proposal_ids or [],
            self_confidence=confidence,
        )

    @staticmethod
    def build_memory_proposal(
        agent_id: str,
        task_id: str,
        thoughts: str | dict[str, Any],
        action_description: str,
        result_observation: str,
        title: str = "New Research Note",
        memory_type: str = "research_note",
        parent_ids: list[str] | None = None,
    ) -> str:
        """
        构建一个 Memory Proposal 并返回 JSON 字符串。
        """
        proposal = ProposalBuilder.create_proposal(
            agent_id=agent_id,
            task_id=task_id,
            thoughts={"abstract": thoughts, "decision_points": [], "considerations": []},
            actions=[{"action_id": "action_1", "description": action_description}],
            observations=[{"observation_id": "obs_1", "description": result_observation}],
            data=[{"data_id": "result_1", "content": result_observation}],
            title=title,
            memory_type=memory_type,
            parent_proposal_ids=parent_ids,
        )
        return proposal.model_dump_json(indent=2)

    @staticmethod
    def self_verify_proposal(
        proposal: MemoryProposal,
        veracity_check: bool = True,
        rationality_check: bool = True,
        value_check: bool = True,
        security_check: bool = True,
        confidence: float = 0.0,
        rationale: str = "",
    ) -> MemoryProposal:
        """
        对提案进行自验证
        """
        proposal.verification.self_verification.veracity = 1 if veracity_check else 0
        proposal.verification.self_verification.rationality = 1 if rationality_check else 0
        proposal.verification.self_verification.value = 1 if value_check else 0
        proposal.verification.self_verification.security = 1 if security_check else 0
        proposal.verification.self_verification.confidence = confidence
        proposal.verification.self_verification.rationale = rationale

        return proposal

    @staticmethod
    def format_proposal_for_agent(proposal: MemoryProposal) -> str:
        """
        将提案格式化为适合Agent处理的字符串格式
        """
        thoughts_str = json.dumps(proposal.body.thoughts, indent=2)
        actions_str = json.dumps(proposal.body.actions, indent=2)
        data_str = json.dumps(proposal.body.data, indent=2)
        obs_str = json.dumps(proposal.body.observations, indent=2)

        return f"""PROPOSAL FOR VERIFICATION:
ID: {proposal.header.proposal_id}
Title: {proposal.header.proposal_summary}
Agent: {proposal.header.proposing_agent_id}
Task: {proposal.header.task_id}
Type: {proposal.header.memory_type}
Timestamp: {proposal.header.timestamp}

THOUGHTS:
{thoughts_str}

ACTIONS:
{actions_str}

DATA:
{data_str}

OBSERVATIONS:
{obs_str}

CURRENT SELF-VERIFICATION:
Veracity: {proposal.verification.self_verification.veracity}
Rationality: {proposal.verification.self_verification.rationality}
Value: {proposal.verification.self_verification.value}
Security: {proposal.verification.self_verification.security}
Confidence: {proposal.verification.self_verification.confidence}
Rationale: {proposal.verification.self_verification.rationale}

Please review and adjust the self-verification scores if necessary."""

def create_proposal_creation_toolkit():
    """
    为Agent提供完整的提案构建工具包
    """
    def build_memory_proposal_func(
        agent_id: str,
        task_id: str,
        thoughts: str,
        action_description: str,
        result_observation: str,
        title: str = "New Research Note",
        memory_type: str = "research_note",
        parent_ids: list[str] | None = None,
    ) -> str:
        """构建一个 Memory Proposal（工具函数）"""
        return ProposalBuilder.build_memory_proposal(
            agent_id=agent_id,
            task_id=task_id,
            thoughts=thoughts,
            action_description=action_description,
            result_observation=result_observation,
            title=title,
            memory_type=memory_type,
            parent_ids=parent_ids,
        )

    def self_verify_proposal_func(
        proposal_json: str,
        veracity: bool = True,
        rationality: bool = True,
        value: bool = True,
        security: bool = True,
        confidence: float = 0.8,
        rationale: str = "Initial self-verification",
    ) -> str:
        """
        对提案进行自验证
        """
        proposal = MemoryProposal.model_validate_json(proposal_json)
        verified = ProposalBuilder.self_verify_proposal(
            proposal,
            veracity_check=veracity,
            rationality_check=rationality,
            value_check=value,
            security_check=security,
            confidence=confidence,
            rationale=rationale,
        )
        return verified.model_dump_json(indent=2)

    def prepare_proposal_for_submission_func(
        agent_id: str,
        task_id: str,
        current_thoughts: str,
        current_action: str,
        current_results: str,
        title: str = "",
        memory_type: str = "research_note",
        confidence: float = 0.8,
        verification_rationale: str = "",
    ) -> str:
        """
        完整的提案准备流程：创建 -> 自验证 -> 准备提交
        这是Agent在完成一轮ReAct循环后应该调用的主要函数
        """
        proposal_json = build_memory_proposal_func(
            agent_id=agent_id,
            task_id=task_id,
            thoughts=current_thoughts,
            action_description=current_action,
            result_observation=current_results,
            title=title,
            memory_type=memory_type,
        )
        rationale = verification_rationale or f"Self-verification of proposal by agent {agent_id}"
        return self_verify_proposal_func(
            proposal_json=proposal_json,
            veracity=True,
            rationality=True,
            value=True,
            security=True,
            confidence=confidence,
            rationale=rationale,
        )

    return {
        "prepare_proposal_for_submission": prepare_proposal_for_submission_func,
        "build_memory_proposal": build_memory_proposal_func,
        "self_verify_proposal": self_verify_proposal_func,
        "format_proposal_for_agent": ProposalBuilder.format_proposal_for_agent,
    }


class MemoryProposalTool:
    """
    用于帮助Agent构建Memory Proposal的工具类
    包含信息整合、自验证等功能
    """

    def __init__(self, orchestrator: Any | None = None):
        """
        Initialize the MemoryProposalTool.

        Args:
            orchestrator: The orchestrator instance to submit proposals to.
                         If None, only prepares proposals for submission.
        """
        self.orchestrator = orchestrator

    def create_proposal(
        self,
        agent_id: str,
        task_id: str,
        thoughts: str | dict[str, Any],
        actions: str | list[dict[str, Any]],
        observations: str | list[dict[str, Any]],
        data: str | list[dict[str, Any]] | dict[str, Any] | None = None,
        title: str = "New Research Note",
        memory_type: str = "research_note",
        parent_ids: list[str] | None = None,
        confidence: float = 0.7,
        veracity: bool = True,
        rationality: bool = True,
        value: bool = True,
        security: bool = True,
        verification_rationale: str = "",
    ) -> str:
        """
        Create a complete memory proposal with self-verification.
        """
        proposal = ProposalBuilder.create_proposal(
            agent_id=agent_id,
            task_id=task_id,
            thoughts=thoughts,
            actions=actions,
            observations=observations,
            data=data,
            title=title,
            memory_type=memory_type,
            parent_proposal_ids=parent_ids,
            confidence=confidence,
        )

        rationale = verification_rationale or f"Self-verification by agent {agent_id} for task {task_id}"
        ProposalBuilder.self_verify_proposal(
            proposal,
            veracity_check=veracity,
            rationality_check=rationality,
            value_check=value,
            security_check=security,
            confidence=confidence,
            rationale=rationale,
        )

        return proposal.model_dump_json(indent=2)

    def submit_proposal(self, proposal_json: str) -> str:
        """
        Submit a prepared proposal to the orchestrator.
        """
        if not self.orchestrator:
            return "No orchestrator available. Proposal prepared but not submitted."

        try:
            proposal_data = json.loads(proposal_json)
            proposal = MemoryProposal(**proposal_data)
            decision = self.orchestrator.verify_and_commit(proposal)
            return f"Proposal {proposal.header.proposal_id} submitted. Decision: {decision.status.value}. Rationale: {decision.rationale}"
        except Exception as e:
            return f"Error submitting proposal: {str(e)}"

    def create_and_submit_proposal(
        self,
        agent_id: str,
        task_id: str,
        thoughts: str | dict[str, Any],
        actions: str | list[dict[str, Any]],
        observations: str | list[dict[str, Any]],
        data: str | list[dict[str, Any]] | dict[str, Any] | None = None,
        title: str = "New Research Note",
        memory_type: str = "research_note",
        parent_ids: list[str] | None = None,
        confidence: float = 0.7,
        veracity: bool = True,
        rationality: bool = True,
        value: bool = True,
        security: bool = True,
        verification_rationale: str = "",
    ) -> str:
        """
        Create and submit a proposal in one step.
        """
        proposal_json = self.create_proposal(
            agent_id=agent_id,
            task_id=task_id,
            thoughts=thoughts,
            actions=actions,
            observations=observations,
            data=data,
            title=title,
            memory_type=memory_type,
            parent_ids=parent_ids,
            confidence=confidence,
            veracity=veracity,
            rationality=rationality,
            value=value,
            security=security,
            verification_rationale=verification_rationale,
        )
        return self.submit_proposal(proposal_json)