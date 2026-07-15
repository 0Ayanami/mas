"""Consensus-memory multi-agent research framework."""

__all__ = [
    "AgentWeightState",
    "AutoGenProposalEvaluator",
    "MajorityVoteConsensus",
    "SmartQuorumConsensus",
    "TAMASAutoGenRunner",
    "TAMASRunConfig",
    "VerificationEngine",
]


def __getattr__(name):
    if name in {
        "AgentWeightState",
        "AutoGenProposalEvaluator",
        "MajorityVoteConsensus",
        "SmartQuorumConsensus",
        "VerificationEngine",
    }:
        from mas_framework import consensus

        return getattr(consensus, name)
    if name in {"TAMASAutoGenRunner", "TAMASRunConfig"}:
        from mas_framework import tamas_workflow

        return getattr(tamas_workflow, name)
    raise AttributeError(f"module 'mas_framework' has no attribute {name!r}")
