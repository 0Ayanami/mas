"""WBFT baseline over AutoGen agent outputs."""

from mas_framework.wbft.consensus import (
    WBFTConsensus,
    parse_wbft_response,
)
from mas_framework.wbft.models import (
    WBFTAgentResponse,
    WBFTAgentSpec,
    WBFTConsensusResult,
    WBFTRunConfig,
)

__all__ = [
    "WBFTAgentResponse",
    "WBFTAgentSpec",
    "WBFTConsensus",
    "WBFTConsensusResult",
    "WBFTRunConfig",
    "WBFTRunResult",
    "WBFTRunner",
    "parse_wbft_response",
]


def __getattr__(name):
    if name in {"WBFTRunResult", "WBFTRunner"}:
        from mas_framework.wbft import autogen_runner

        return getattr(autogen_runner, name)
    raise AttributeError(f"module 'mas_framework.wbft' has no attribute {name!r}")
