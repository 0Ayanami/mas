"""Agent state and weight management for memory consensus research."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Union

from fisher_lda import (
    FisherLDAResult,
    RawSample,
    fit_fisher_lda_quality_weights,
)


def _bounded_score(name: str, value: float) -> float:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return float(value)


@dataclass
class AgentWeightState:
    """Consensus-independent state used to derive an agent's current weight."""

    agent_id: str
    capability_coefficient: float = 1.0
    initial_vc: float = 0.5
    initial_hc: float = 0.5
    proposal_confidences: Deque[float] = field(default_factory=deque)
    vote_alignments: Deque[bool] = field(default_factory=deque)

    @property
    def verified_confidence(self) -> float:
        # 采用滑动窗口保存最近K(20)个proposal的confidence score
        if not self.proposal_confidences:
            return self.initial_vc
        return sum(self.proposal_confidences) / len(self.proposal_confidences)

    @property
    def historical_confidence(self) -> float:
        # 采用滑动窗口保存最近K(30)个vote的alignment score
        if not self.vote_alignments:
            return self.initial_hc
        return sum(1.0 for aligned in self.vote_alignments if aligned) / len(
            self.vote_alignments
        )

    def snapshot(
        self,
        *,
        alpha: float,
        beta: float,
        gamma: float,
        theta: float,
    ) -> Dict[str, Union[float, str, int]]:
        quality = alpha * self.verified_confidence + beta * self.historical_confidence
        weight = self.capability_coefficient * math.exp(gamma * (quality - theta))
        return {
            "agent_id": self.agent_id,
            "verified_confidence": self.verified_confidence,
            "historical_confidence": self.historical_confidence,
            "quality": quality,
            "weight": weight,
            "capability_coefficient": self.capability_coefficient,
            "proposal_samples": len(self.proposal_confidences),
            "vote_samples": len(self.vote_alignments),
        }


class WeightManager:
    """Maintains per-agent state used by future consensus algorithms."""

    def __init__(
        self,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        gamma: float = 5.0,
        theta: float = 0.5,
        proposal_window: int = 20,
        vote_window: int = 30,
        initial_vc: float = 0.5,
        initial_hc: float = 0.5,
    ) -> None:
        if abs((alpha + beta) - 1.0) > 1e-6:
            raise ValueError("alpha and beta must sum to 1.")
        if proposal_window <= 0 or vote_window <= 0:
            raise ValueError("Window sizes must be positive.")
        self.alpha = _bounded_score("alpha", alpha)
        self.beta = _bounded_score("beta", beta)
        self.gamma = float(gamma)
        self.theta = _bounded_score("theta", theta)
        self.proposal_window = proposal_window
        self.vote_window = vote_window
        self.initial_vc = _bounded_score("initial_vc", initial_vc)
        self.initial_hc = _bounded_score("initial_hc", initial_hc)
        self._states: Dict[str, AgentWeightState] = {}

    def register_agent(
        self, agent_id: str, capability_coefficient: float = 1.0
    ) -> AgentWeightState:
        """Register or return an agent state."""
        if capability_coefficient <= 0:
            raise ValueError("capability_coefficient must be positive.")
        if agent_id not in self._states:
            self._states[agent_id] = AgentWeightState(
                agent_id=agent_id,
                capability_coefficient=capability_coefficient,
                initial_vc=self.initial_vc,
                initial_hc=self.initial_hc,
            )
        return self._states[agent_id]

    def ensure_agents(self, agent_ids: Iterable[str]) -> None:
        for agent_id in agent_ids:
            self.register_agent(agent_id)

    def get_state(self, agent_id: str) -> AgentWeightState:
        return self.register_agent(agent_id)

    def update_capability(
        self, agent_id: str, capability_coefficient: float
    ) -> AgentWeightState:
        if capability_coefficient <= 0:
            raise ValueError("capability_coefficient must be positive.")
        state = self.register_agent(agent_id)
        state.capability_coefficient = capability_coefficient
        return state

    def record_proposal_confidence(
        self, agent_id: str, confidence_score: float
    ) -> AgentWeightState:
        """Record post-verification confidence supplied by an external workflow."""
        state = self.register_agent(agent_id)
        state.proposal_confidences.append(
            _bounded_score("confidence_score", confidence_score)
        )
        self._trim_left(state.proposal_confidences, self.proposal_window)
        return state

    def record_vote_alignment(
        self, agent_id: str, aligned_with_outcome: bool
    ) -> AgentWeightState:
        """Record whether an agent's vote aligned with an external final outcome."""
        state = self.register_agent(agent_id)
        state.vote_alignments.append(bool(aligned_with_outcome))
        self._trim_left(state.vote_alignments, self.vote_window)
        return state

    def verified_confidence(self, agent_id: str) -> float:
        return self.get_state(agent_id).verified_confidence

    def historical_confidence(self, agent_id: str) -> float:
        return self.get_state(agent_id).historical_confidence

    def quality(self, agent_id: str) -> float:
        state = self.get_state(agent_id)
        return (
            self.alpha * state.verified_confidence
            + self.beta * state.historical_confidence
        )

    def weight(self, agent_id: str) -> float:
        state = self.get_state(agent_id)
        return state.capability_coefficient * math.exp(
            self.gamma * (self.quality(agent_id) - self.theta)
        )

    def snapshot(self, agent_id: str) -> Dict[str, Union[float, str, int]]:
        return self.get_state(agent_id).snapshot(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            theta=self.theta,
        )

    def snapshots(self) -> List[Dict[str, Union[float, str, int]]]:
        return [self.snapshot(agent_id) for agent_id in sorted(self._states)]

    def fit_quality_weights(
        self,
        samples: Iterable[RawSample],
        *,
        update_theta: bool = False,
        regularization: float = 1e-6,
    ) -> FisherLDAResult:
        """Fit alpha/beta from labeled vc/hc samples using Fisher LDA."""
        result = fit_fisher_lda_quality_weights(
            samples,
            regularization=regularization,
        )
        self.alpha = result.alpha
        self.beta = result.beta
        if update_theta:
            self.theta = result.theta
        return result

    def _trim_left(self, values: Deque[object], max_length: int) -> None:
        while len(values) > max_length:
            values.popleft()
