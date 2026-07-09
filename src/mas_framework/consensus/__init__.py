"""Consensus-memory infrastructure for MARBLE."""

from marble.memory.consensus_memory import ConsensusMemory
from marble.consensus.models import (
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
    SelfVerificationScores,
    VerificationContext,
    VerificationVector,
)
from marble.consensus.majority_vote import MajorityVoteConsensus
from marble.consensus.fisher_lda import (
    FisherLDAResult,
    FisherLDASample,
    fit_fisher_lda_quality_weights,
)
from marble.consensus.smart_quorum import SmartQuorumConsensus
from marble.consensus.layer import (
    AgentActivityTracker,
    ConsensusLayer,
    ProposalSubmission,
)
from marble.consensus.proposal_builder import (
    HashSignatureProvider,
    ProposalBuilder,
    SignatureProvider,
)
from marble.consensus.verification_engine import (
    HeuristicProposalEvaluator,
    LLMProposalEvaluator,
    ProposalEvaluator,
    VerificationEngine,
    load_consensus_env,
)
from marble.consensus.weight_manager import AgentWeightState, WeightManager
from marble.consensus.workflow import (
    ConsensusWorkflowResult,
    MemoryConsensusWorkflow,
)

__all__ = [
    "AgentWeightState",
    "AgentActivityTracker",
    "ConsensusMemory",
    "ConsensusDecision",
    "ConsensusLayer",
    "ConsensusResult",
    "ConsensusVote",
    "ConsensusWorkflowResult",
    "DEFAULT_DIMENSION_WEIGHTS",
    "FisherLDAResult",
    "FisherLDASample",
    "HashSignatureProvider",
    "HeuristicProposalEvaluator",
    "KeyDecision",
    "LLMProposalEvaluator",
    "MemoryProposal",
    "MemoryProposalBody",
    "MemoryProposalHeader",
    "MultiVerificationSummary",
    "MajorityVoteConsensus",
    "MemoryConsensusWorkflow",
    "ProposalAction",
    "ProposalBuilder",
    "ProposalDataReference",
    "ProposalEvaluator",
    "ProposalObservation",
    "ProposalSubmission",
    "ProposalThoughts",
    "ProposalVerification",
    "SelfVerificationScores",
    "SignatureProvider",
    "SmartQuorumConsensus",
    "VERIFICATION_DIMENSIONS",
    "VerificationContext",
    "VerificationEngine",
    "VerificationVector",
    "WeightManager",
    "fit_fisher_lda_quality_weights",
    "load_consensus_env",
]
