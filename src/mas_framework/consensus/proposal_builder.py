"""Builder for standardized MARBLE memory proposals."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Union

from mas_framework.consensus.models import (
    KeyDecision,
    MemoryProposal,
    MemoryProposalBody,
    MemoryProposalHeader,
    ProposalAction,
    ProposalDataReference,
    ProposalObservation,
    ProposalThoughts,
    ProposalVerification,
    SelfVerificationScores,
    canonical_json,
    utc_now_iso,
)


class SignatureProvider(Protocol):
    """Interface for proposal body signing."""

    def sign(self, agent_id: str, body_hash: str, body_json: str) -> str:
        """Return a signature for the canonical proposal body."""


class HashSignatureProvider:
    """Deterministic local signer used when no cryptographic signer is supplied."""

    def sign(self, agent_id: str, body_hash: str, body_json: str) -> str:
        payload = f"{agent_id}:{body_hash}:{body_json}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class ProposalBuilder:
    """Constructs standardized memory proposals from MARBLE agent outputs."""

    def __init__(self, signature_provider: Optional[SignatureProvider] = None) -> None:
        self.signature_provider = signature_provider or HashSignatureProvider()

    def build(
        self,
        *,
        task_id: str,
        agent_id: str,
        thoughts: Optional[Union[ProposalThoughts, Dict[str, Any], str]] = None,
        actions: Optional[Iterable[Union[ProposalAction, Dict[str, Any]]]] = None,
        data: Optional[Iterable[Union[ProposalDataReference, Dict[str, Any]]]] = None,
        observations: Optional[
            Iterable[Union[ProposalObservation, Dict[str, Any], str]]
        ] = None,
        self_verification: Optional[
            Union[SelfVerificationScores, Dict[str, float]]
        ] = None,
        parent_proposals: Optional[Sequence[str]] = None,
        proposal_summary: Optional[str] = None,
        timestamp: Optional[str] = None,
        proposal_id: Optional[str] = None,
    ) -> MemoryProposal:
        """Build a proposal without making any acceptance decision."""
        body = MemoryProposalBody(
            thoughts=self._coerce_thoughts(thoughts),
            actions=self._coerce_actions(actions or []),
            data=self._coerce_data(data or []),
            observations=self._coerce_observations(observations or []),
        )
        if body.is_empty():
            raise ValueError("A memory proposal body must contain at least one item.")

        canonical_body = canonical_json(body)
        body_hash = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
        signature = self.signature_provider.sign(agent_id, body_hash, canonical_body)
        summary = self._summary_or_default(proposal_summary, body)

        header = MemoryProposalHeader(
            proposal_id=proposal_id or str(uuid.uuid4()),
            task_id=task_id,
            timestamp=timestamp or utc_now_iso(),
            agent_id=agent_id,
            agent_signature=signature,
            parent_proposals=list(parent_proposals or []),
            body_hash=body_hash,
            proposal_summary=summary,
        )
        verification = ProposalVerification(
            self_verification=self._coerce_self_verification(self_verification)
        )
        return MemoryProposal(header=header, body=body, verification=verification)

    def from_agent_output(
        self,
        *,
        task_id: str,
        agent_id: str,
        output: Any,
        self_verification: Optional[
            Union[SelfVerificationScores, Dict[str, float]]
        ] = None,
        proposal_summary: Optional[str] = None,
        parent_proposals: Optional[Sequence[str]] = None,
    ) -> MemoryProposal:
        """Create a proposal from a raw MARBLE agent result."""
        if isinstance(output, dict):
            return self.build(
                task_id=task_id,
                agent_id=agent_id,
                thoughts=output.get("thoughts"),
                actions=output.get("actions"),
                data=output.get("data"),
                observations=output.get("observations")
                or [{"type": "agent_output", "description": str(output)}],
                self_verification=self_verification,
                parent_proposals=parent_proposals,
                proposal_summary=proposal_summary or output.get("proposal_summary"),
            )

        return self.build(
            task_id=task_id,
            agent_id=agent_id,
            observations=[{"type": "agent_output", "description": str(output)}],
            self_verification=self_verification,
            parent_proposals=parent_proposals,
            proposal_summary=proposal_summary,
        )

    def _coerce_thoughts(
        self, thoughts: Optional[Union[ProposalThoughts, Dict[str, Any], str]]
    ) -> Optional[ProposalThoughts]:
        if thoughts is None:
            return None
        if isinstance(thoughts, ProposalThoughts):
            return thoughts
        if isinstance(thoughts, str):
            return ProposalThoughts(thoughts_abstract=thoughts[:500])

        decisions = [
            decision
            if isinstance(decision, KeyDecision)
            else KeyDecision(
                decision=str(decision.get("decision", "")),
                result=str(decision.get("result", "")),
            )
            for decision in thoughts.get("key_decisions", [])
        ]
        return ProposalThoughts(
            thoughts_abstract=str(thoughts.get("thoughts_abstract", ""))[:500],
            key_decisions=decisions,
        )

    def _coerce_actions(
        self, actions: Iterable[Union[ProposalAction, Dict[str, Any]]]
    ) -> List[ProposalAction]:
        return [
            action
            if isinstance(action, ProposalAction)
            else ProposalAction(
                action_id=action.get("action_id"),
                type=str(action.get("type", "")),
                tool=str(action.get("tool", "")),
                params=dict(action.get("params", {})),
                status=str(action.get("status", "")),
            )
            for action in actions
        ]

    def _coerce_data(
        self, data: Iterable[Union[ProposalDataReference, Dict[str, Any]]]
    ) -> List[ProposalDataReference]:
        return [
            item
            if isinstance(item, ProposalDataReference)
            else ProposalDataReference(
                source=str(item.get("source", "")),
                content_snippet=str(item.get("content_snippet", "")),
                url=str(item.get("url", "")),
                timestamp=str(item.get("timestamp", "")),
            )
            for item in data
        ]

    def _coerce_observations(
        self,
        observations: Iterable[Union[ProposalObservation, Dict[str, Any], str]],
    ) -> List[ProposalObservation]:
        coerced = []
        for observation in observations:
            if isinstance(observation, ProposalObservation):
                coerced.append(observation)
            elif isinstance(observation, str):
                coerced.append(
                    ProposalObservation(type="agent_output", description=observation)
                )
            else:
                coerced.append(
                    ProposalObservation(
                        type=str(observation.get("type", "")),
                        description=str(observation.get("description", "")),
                        status=str(observation.get("status", "")),
                    )
                )
        return coerced

    def _coerce_self_verification(
        self,
        self_verification: Optional[Union[SelfVerificationScores, Dict[str, float]]],
    ) -> SelfVerificationScores:
        if isinstance(self_verification, SelfVerificationScores):
            return self_verification
        if not self_verification:
            return SelfVerificationScores(
                veracity_score=0,
                rationality_score=0,
                value_score=0,
                security_score=0,
            )
        return SelfVerificationScores(
            veracity_score=self_verification["veracity_score"],
            rationality_score=self_verification["rationality_score"],
            value_score=self_verification["value_score"],
            security_score=self_verification["security_score"],
        )

    def _summary_or_default(
        self, proposal_summary: Optional[str], body: MemoryProposalBody
    ) -> str:
        if proposal_summary:
            return proposal_summary[:200]
        if body.observations:
            return body.observations[0].description[:200]
        if body.data:
            return body.data[0].content_snippet[:200]
        if body.actions:
            return f"{body.actions[0].type}:{body.actions[0].tool}"[:200]
        if body.thoughts:
            return body.thoughts.thoughts_abstract[:200]
        return "Memory proposal"
