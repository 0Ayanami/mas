from __future__ import annotations

from statistics import mean

from mas_framework.models import (
    ConsensusDecision,
    ConsensusResult,
    MemoryProposal,
    MultiAgentVerificationSummary,
    ProposalStatus,
)


class SmartQuorumPolicy:
    def __init__(self, confidence_threshold: float = 0.7, base_threshold: float = 2 / 3, min_threshold: float = 0.55, agent_count:int=3):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.confidence_threshold = confidence_threshold
        self.agent_count = agent_count

    def threshold_for(self, proposal: MemoryProposal, validator_count: int) -> float:
        """
        计算当前系统的Quorum阈值
        """
        pass

    def decide(self, proposal: MemoryProposal, validator_count: int) -> ConsensusDecision:
        if not proposal.verifications or validator_count <= 3:
            return ConsensusDecision(
                proposal_id=proposal.proposal_id,
                status=ProposalStatus.REJECTED,
                quorum_size=validator_count,
                positive_votes=0,
                average_confidence=0.0,
                threshold=1.0,
                rationale="No verification votes were collected.",
            )
        
        # 得到每个agent权重列表，并计算总权重
        weights = [getattr(vote, "weight", 1.0) for vote in proposal.verifications]
        total_weight = sum(weights) if weights else float(validator_count)

        positive_weight = sum(
            getattr(vote, "weight", 1.0)
            for vote in proposal.verifications
            if vote.confidence >= self.confidence_threshold
        )
        # 赞成权重占比
        vote_ratio = positive_weight / total_weight if total_weight > 0 else 0.0

        # 计算权重平均值
        def _weighted_mean(attr: str) -> float:
            if total_weight <= 0:
                return mean(getattr(vote, attr) for vote in proposal.verifications)
            return sum(getattr(vote, attr) * getattr(vote, "weight", 1.0) for vote in proposal.verifications) / total_weight
        
        # 得到阈值
        threshold = self.threshold_for(proposal, validator_count)
        accepted = vote_ratio >= threshold

        avg_confidence = round(_weighted_mean("confidence"), 4)

        proposal.verification.multi_agent_verification = MultiAgentVerificationSummary(
            veracity=round(_weighted_mean("veracity"), 4),
            rationality=round(_weighted_mean("rationality"), 4),
            value=round(_weighted_mean("value"), 4),
            security=round(_weighted_mean("security"), 4),
            confidence=avg_confidence,
            verifier_count=len(proposal.verifications),
        )

        proposal.verification.consensus_result = ConsensusResult(
            voting_agents=validator_count,
            total_agents=self.agent_count,
            vote_weight=float(round(positive_weight, 4)),
            total_weight=float(round(total_weight, 4)),
            result=ProposalStatus.ACCEPTED if accepted else ProposalStatus.REJECTED,
        )

        return ConsensusDecision(
            proposal_id=proposal.proposal_id,
            status=ProposalStatus.ACCEPTED if accepted else ProposalStatus.REJECTED,
            quorum_size=validator_count,
            positive_votes=float(round(positive_weight, 4)),
            average_confidence=avg_confidence,
            threshold=round(threshold, 4),
            rationale=(
                f"positive_vote_ratio={vote_ratio:.3f}, "
                f"average_confidence={avg_confidence:.3f}, "
                f"threshold={threshold:.3f}"
            ),
        )
