"""Majority-vote baseline consensus for memory proposals."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from mas_framework.consensus.models import (
    ConsensusDecision,
    ConsensusResult,
    ConsensusVote,
    MemoryProposal,
    MultiVerificationSummary,
    VERIFICATION_DIMENSIONS,
    VerificationVector,
)


class MajorityVoteConsensus:
    """
    Simple majority-vote consensus baseline.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.6,
        majority_threshold: float = 0.5,
        strict_majority: bool = True,
        minimum_votes: int = 1,
        hard_fail_dimensions: Sequence[str] = ("security",),
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= majority_threshold <= 1.0:
            raise ValueError("majority_threshold must be between 0 and 1.")
        if minimum_votes < 1:
            raise ValueError("minimum_votes must be at least 1.")
        
        self.confidence_threshold = confidence_threshold
        self.majority_threshold = majority_threshold
        self.strict_majority = strict_majority
        self.minimum_votes = minimum_votes

        unknown_dimensions = [
            dimension
            for dimension in hard_fail_dimensions
            if dimension not in VERIFICATION_DIMENSIONS
        ]
        if unknown_dimensions:
            raise ValueError(
                "Unknown hard-fail dimensions: " + ", ".join(unknown_dimensions)
            )
        self.hard_fail_dimensions = tuple(hard_fail_dimensions)

    def build_vote(
        self,
        proposal: MemoryProposal,
        verification: VerificationVector,
        voter_agent_id: Optional[str] = None,
    ) -> ConsensusVote:
        """Convert one verification vector into a baseline vote."""
        voter = voter_agent_id or verification.verifier_agent_id
        if not voter:
            raise ValueError("A consensus vote requires a voter_agent_id.")
        
        confidence = verification.confidence_score

        # 要求某些维度必须验证通过
        hard_failures = [
            dimension
            for dimension in self.hard_fail_dimensions
            if getattr(verification, dimension) == 0
        ]
        accept = (
            confidence >= self.confidence_threshold
            and len(hard_failures) == 0
        )
        if hard_failures:
            reason = "hard-fail dimensions: " + ", ".join(hard_failures)
        elif accept:
            reason = f"confidence {confidence:.3f} >= {self.confidence_threshold:.3f}"
        else:
            reason = f"confidence {confidence:.3f} < {self.confidence_threshold:.3f}"
        return ConsensusVote(
            proposal_id=proposal.proposal_id,
            voter_agent_id=voter,
            accept=accept,
            confidence_score=confidence,
            verification=verification,
            reason=reason,
        )

    def decide(
        self,
        proposal: MemoryProposal,
        verifications: Iterable[VerificationVector],
    ) -> ConsensusDecision:
        """Run majority vote over verification vectors for one proposal."""
        # 从验证向量中构建每个agent的投票结果
        votes = [self.build_vote(proposal, vector) for vector in verifications]
        multi_verification = self._average_dimension_summary(votes)
        if len(votes) < self.minimum_votes:
            consensus_result = ConsensusResult(
                total_weight=float(len(votes)),
                vote_weight=float(sum(1 for vote in votes if vote.accept)),
                result="pending",
            )
            return ConsensusDecision(
                proposal_id=proposal.proposal_id,
                result="pending",
                votes=votes,
                accept_count=sum(1 for vote in votes if vote.accept),
                reject_count=sum(1 for vote in votes if not vote.accept),
                threshold=self.majority_threshold,
                confidence_threshold=self.confidence_threshold,
                total_weight=float(len(votes)),
                accept_weight=float(sum(1 for vote in votes if vote.accept)),
                metadata={
                    "strategy": "majority_vote",
                    "reason": "minimum_votes_not_met",
                    "proposal_confidence_score": 0.0,
                    "proposal_confidence_method": "arithmetic_mean",
                    "multi_verification_summary": multi_verification.to_dict(),
                    "consensus_result": consensus_result.to_dict(),
                },
            )

        accept_count = sum(1 for vote in votes if vote.accept)
        reject_count = len(votes) - accept_count
        acceptance_ratio = accept_count / len(votes)
        if self.strict_majority:
            passed = acceptance_ratio > self.majority_threshold
        else:
            passed = acceptance_ratio >= self.majority_threshold
        proposal_confidence_score = (
            self._average_confidence(votes) if passed else 0.0
        )
        consensus_result = ConsensusResult(
            total_weight=float(len(votes)),
            vote_weight=float(accept_count),
            result="pass" if passed else "fail",
        )

        return ConsensusDecision(
            proposal_id=proposal.proposal_id,
            result="pass" if passed else "fail",
            votes=votes,
            accept_count=accept_count,
            reject_count=reject_count,
            acceptance_ratio=acceptance_ratio,
            threshold=self.majority_threshold,
            confidence_threshold=self.confidence_threshold,
            total_weight=float(len(votes)),
            accept_weight=float(accept_count),
            metadata={
                "strategy": "majority_vote",
                "proposal_confidence_score": proposal_confidence_score,
                "proposal_confidence_method": "arithmetic_mean",
                "multi_verification_summary": multi_verification.to_dict(),
                "consensus_result": consensus_result.to_dict(),
            },
        )

    def _average_confidence(self, votes: Sequence[ConsensusVote]) -> float:
        if not votes:
            return 0.0
        return sum(vote.confidence_score for vote in votes) / len(votes)

    def _average_dimension_summary(
        self,
        votes: Sequence[ConsensusVote],
    ) -> MultiVerificationSummary:
        if not votes:
            return MultiVerificationSummary(
                weighted_scores={dimension: 0.0 for dimension in VERIFICATION_DIMENSIONS}
            )
        return MultiVerificationSummary(
            weighted_scores={
                dimension: sum(
                    getattr(vote.verification, dimension)
                    for vote in votes
                )
                / len(votes)
                for dimension in VERIFICATION_DIMENSIONS
            }
        )
