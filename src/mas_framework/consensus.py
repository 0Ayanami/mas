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
    def __init__(self, base_threshold: float = 2 / 3, min_threshold: float = 0.55):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold

    def threshold_for(self, proposal: MemoryProposal, validator_count: int) -> float:
        pass

    def decide(self, proposal: MemoryProposal, agent_count: int) -> ConsensusDecision:
        validator_count = len(proposal.verifications)
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

        # 计算权重平均值
        def _weighted_mean(attr: str) -> float:
            if total_weight <= 0:
                return mean(getattr(vote, attr) for vote in proposal.verifications)
            return round(
                sum(getattr(vote, attr) * getattr(vote, "weight", 1.0) for vote in proposal.verifications)
                / total_weight,
                4,
            )

        threshold = self.threshold_for(proposal, validator_count)
        accepted = vote_ratio >= threshold
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

        proposal.verification.consensus_result = ConsensusResult(
            voting_agents=validator_count,
            total_agents=agent_count,
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
