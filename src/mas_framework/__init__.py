"""Consensus-memory multi-agent research framework."""

from mas_framework.models import (
    AgentConfig,
    ConsensusDecision,
    MemoryProposal,
    ProposalStatus,
    VerificationVector,
    AgentState,
    SelfVerification,
)
from mas_framework.orchestrator import Orchestrator as ResearchOrchestrator

__all__ = [
    "AgentConfig",
    "AgentState",
    "ConsensusDecision",
    "MemoryProposal",
    "ProposalStatus",
    "ResearchOrchestrator",
    "SelfVerification",
    "VerificationVector",
]
