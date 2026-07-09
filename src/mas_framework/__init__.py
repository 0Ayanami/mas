"""Consensus-memory multi-agent research framework."""

from mas_framework.consensus import (
    AgentWeightState,
    AutoGenProposalEvaluator,
    MajorityVoteConsensus,
    SmartQuorumConsensus,
    VerificationEngine,
)
from mas_framework.common import AgentConfig
from mas_framework.tamas_workflow import TAMASAutoGenRunner, TAMASRunConfig

__all__ = [
    "AgentConfig",
    "AgentWeightState",
    "AutoGenProposalEvaluator",
    "MajorityVoteConsensus",
    "SmartQuorumConsensus",
    "TAMASAutoGenRunner",
    "TAMASRunConfig",
    "VerificationEngine",
]
