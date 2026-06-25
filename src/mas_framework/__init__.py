"""Consensus-memory multi-agent research framework."""

from mas_framework.models import (
    AgentConfig,
    ConsensusResult,
    MemoryProposal,
    ProposalStatus,
    VerificationVector,
    AgentState,
    SelfVerification,
)
from mas_framework.orchestrator import Orchestrator as ResearchOrchestrator
from mas_framework.agentdojo_adapter import AgentDojoMASPipeline

__all__ = [
    "AgentDojoMASPipeline",
    "AgentConfig",
    "AgentState",
    "ConsensusResult",
    "MemoryProposal",
    "ProposalStatus",
    "ResearchOrchestrator",
    "SelfVerification",
    "VerificationVector",
]
