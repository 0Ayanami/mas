from mas_framework.consensus.majority_vote import MajorityVoteConsensus
from mas_framework.consensus.models import (
    DEFAULT_DIMENSION_WEIGHTS,
    VERIFICATION_DIMENSIONS,
    ConsensusDecision,
    ConsensusResult,
    ConsensusVote,
    KeyDecision,
    MemoryProposal,
    MemoryProposalBody,
    MemoryProposalHeader,
    MultiVerificationSummary,
    ProposalAction,
    ProposalDataReference,
    ProposalObservation,
    ProposalThoughts,
    ProposalVerification,
    VerificationContext,
    VerificationVector,
)
from mas_framework.consensus.proposal_builder import HashSignatureProvider, ProposalBuilder, SignatureProvider
from mas_framework.consensus.smart_quorum import SmartQuorumConsensus
from mas_framework.consensus.verification_engine import (
    AutoGenProposalEvaluator,
    HeuristicProposalEvaluator,
    ProposalEvaluator,
    VerificationEngine,
)
from mas_framework.consensus.weight_manager import AgentWeightState, WeightManager

MajorityVotePolicy = MajorityVoteConsensus
SmartQuorumPolicy = SmartQuorumConsensus
WeightedQuorumPolicy = SmartQuorumConsensus

__all__ = [
    "AgentWeightState",
    "AutoGenProposalEvaluator",
    "ConsensusDecision",
    "ConsensusResult",
    "ConsensusVote",
    "DEFAULT_DIMENSION_WEIGHTS",
    "HashSignatureProvider",
    "HeuristicProposalEvaluator",
    "KeyDecision",
    "MajorityVoteConsensus",
    "MajorityVotePolicy",
    "MemoryProposal",
    "MemoryProposalBody",
    "MemoryProposalHeader",
    "MultiVerificationSummary",
    "ProposalAction",
    "ProposalBuilder",
    "ProposalDataReference",
    "ProposalEvaluator",
    "ProposalObservation",
    "ProposalThoughts",
    "ProposalVerification",
    "SignatureProvider",
    "SmartQuorumConsensus",
    "SmartQuorumPolicy",
    "VERIFICATION_DIMENSIONS",
    "VerificationContext",
    "VerificationEngine",
    "VerificationVector",
    "WeightManager",
    "WeightedQuorumPolicy",
]
