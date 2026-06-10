"""Consensus-memory multi-agent research framework."""

from mas_framework.models import (
    AgentConfig,
    ConsensusDecision,
    MemoryProposal,
    ProposalStatus,
    VerificationVector,
)
from mas_framework.orchestrator import Orchestrator as ResearchOrchestrator

__all__ = [
    "AgentConfig",
    "ConsensusDecision",
    "MemoryProposal",
    "ProposalStatus",
    "ResearchOrchestrator",
    "VerificationVector",
]

