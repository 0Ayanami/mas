from __future__ import annotations

from statistics import mean

from mas_framework.models import (
    ConsensusResult,
    MemoryProposal,
    MultiAgentVerificationSummary,
    ProposalStatus,
)


class SmartQuorumPolicy:
    def __init__(self, base_threshold: float = 2 / 3, min_threshold: float = 0.55):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold

    def threshold_for(self, proposal: MemoryProposal, validator_count: int) -> float | None:
        pass

    def decide(self, proposal: MemoryProposal, agent_count: int) -> ConsensusResult:
        validator_count = len(proposal.verifications)
        # 决策时需要确保参与共识的agent数大于等于3
        if validator_count <= 3:
            return ConsensusResult(
            voting_agents=validator_count,
            total_agents=agent_count,
            vote_weight=0.0,
            total_weight=0.0,
            result=ProposalStatus.REJECTED,
        )
        
        # 计算总权重
        total_weight = sum(getattr(vote, "weight", 1.0) for vote in proposal.verifications)
        # 计算赞成权重
        positive_weight = sum(
            getattr(vote, "weight", 1.0)
            for vote in proposal.verifications
            if vote.vote_result
        )
        # 赞成权重占比
        vote_ratio = round(positive_weight / total_weight, 4) if total_weight > 0 else 0.0

        # 根据阈值判断是否通过
        threshold = self.threshold_for(proposal, validator_count)
        if threshold is None:
            threshold = self.base_threshold
        accepted = vote_ratio >= threshold
        
        # 计算权重平均值
        def _weighted_mean(attr: str) -> float:
            return round(
                sum(getattr(vote, attr) * getattr(vote, "weight", 1.0) for vote in proposal.verifications)
                / total_weight,
                4,
            )

        avg_confidence = _weighted_mean("confidence")

        # Update proposal verification summary
        proposal.verification.multi_agent_verification = MultiAgentVerificationSummary(
            veracity=_weighted_mean("veracity"),
            rationality=_weighted_mean("rationality"),
            value=_weighted_mean("value"),
            security=_weighted_mean("security"),
            confidence=avg_confidence,
            verifier_count=validator_count,
        )
        return ConsensusResult(
            voting_agents=validator_count,
            total_agents=agent_count,
            vote_weight=float(round(positive_weight, 4)),
            total_weight=float(round(total_weight, 4)),
            result=ProposalStatus.ACCEPTED if accepted else ProposalStatus.REJECTED,
        )
