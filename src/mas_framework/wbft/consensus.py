"""Confidence-weighted WBFT consensus over agent task outputs."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from mas_framework.wbft.models import (
    WBFTAgentResponse,
    WBFTConsensusResult,
)


WBFT_RESPONSE_MARKER = "WBFT_RESPONSE"
WBFT_RESPONSE_END = "END_WBFT_RESPONSE"


class WBFTConsensus:
    """
    Agent-output WBFT baseline.

    This mirrors the source WBFT project's content-agnostic idea: agents produce
    answers and confidence values, and the protocol picks a final answer using
    confidence-weighted voting plus Byzantine-safety trace metadata.
    """

    algorithm_name = "confidence_weighted_wbft"

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.0,
        convergence_threshold: float = 0.0,
        fault_tolerance_threshold: float = 0.33,
        minimum_participants: int = 1,
        normalization: str = "auto",
    ) -> None:
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= float(fault_tolerance_threshold) <= 1.0:
            raise ValueError("fault_tolerance_threshold must be between 0 and 1.")
        if minimum_participants < 1:
            raise ValueError("minimum_participants must be at least 1.")
        if normalization not in {"auto", "number", "none"}:
            raise ValueError("normalization must be auto, number, or none.")

        self.confidence_threshold = float(confidence_threshold)
        self.convergence_threshold = float(convergence_threshold)
        self.fault_tolerance_threshold = float(fault_tolerance_threshold)
        self.minimum_participants = int(minimum_participants)
        self.normalization = normalization

    def decide(self, responses: list[WBFTAgentResponse]) -> WBFTConsensusResult:
        if len(responses) < self.minimum_participants:
            return WBFTConsensusResult(
                consensus_answer="",
                consensus_confidence=0.0,
                participant_count=len(responses),
                convergence_achieved=False,
                responses=responses,
                metadata={
                    "algorithm": self.algorithm_name,
                    "reason": "minimum_participants_not_met",
                    "minimum_participants": self.minimum_participants,
                },
            )

        grouped: dict[str, list[WBFTAgentResponse]] = defaultdict(list)
        answer_by_normalized: dict[str, str] = {}
        for response in responses:
            normalized = self.normalize_answer(response.answer)
            grouped[normalized].append(response)
            answer_by_normalized.setdefault(normalized, response.answer)

        raw_confidences = {
            answer: [response.confidence for response in answer_responses]
            for answer, answer_responses in grouped.items()
        }
        filtered_confidences = {
            answer: [
                confidence
                for confidence in confidences
                if confidence >= self.confidence_threshold
            ]
            or confidences[:]
            for answer, confidences in raw_confidences.items()
        }
        average_confidence_by_answer = {
            answer: (
                sum(confidences) / len(confidences)
                if confidences
                else 0.0
            )
            for answer, confidences in filtered_confidences.items()
        }
        support_count_by_answer = {
            answer: len(confidences)
            for answer, confidences in filtered_confidences.items()
        }
        total_confidence_weight = sum(
            sum(confidences) for confidences in raw_confidences.values()
        )

        if total_confidence_weight == 0.0:
            answer_counts = Counter(
                self.normalize_answer(response.answer) for response in responses
            )
            winning_answer, vote_count = answer_counts.most_common(1)[0]
            consensus_confidence = vote_count / len(responses)
            confidence_weight_ratio = consensus_confidence
            scoring_method = "fallback_majority"
        else:
            winning_answer = max(
                average_confidence_by_answer,
                key=lambda answer: (
                    average_confidence_by_answer[answer],
                    support_count_by_answer[answer],
                ),
            )
            vote_count = support_count_by_answer[winning_answer]
            consensus_confidence = average_confidence_by_answer[winning_answer]
            confidence_weight_ratio = consensus_confidence
            scoring_method = "average_confidence_priority"

        total_nodes = len(responses)
        max_faulty_nodes = int(total_nodes * self.fault_tolerance_threshold)
        byzantine_safe = self._check_byzantine_safety(
            vote_count=vote_count,
            total_nodes=total_nodes,
            max_faulty=max_faulty_nodes,
        )
        convergence_achieved = confidence_weight_ratio >= (
            0.5 + self.convergence_threshold
        )

        return WBFTConsensusResult(
            consensus_answer=answer_by_normalized.get(winning_answer, winning_answer),
            consensus_confidence=consensus_confidence,
            participant_count=total_nodes,
            convergence_achieved=convergence_achieved,
            responses=responses,
            metadata={
                "algorithm": self.algorithm_name,
                "scoring_method": scoring_method,
                "consensus_method": "confidence_priority_consensus",
                "normalization": self.normalization,
                "normalized_consensus_answer": winning_answer,
                "confidence_threshold": self.confidence_threshold,
                "fault_tolerance_threshold": self.fault_tolerance_threshold,
                "total_nodes": total_nodes,
                "max_faulty_nodes": max_faulty_nodes,
                "vote_count": vote_count,
                "answer_distribution": {
                    answer: len(answer_responses)
                    for answer, answer_responses in grouped.items()
                },
                "raw_answer_confidences": raw_confidences,
                "filtered_answer_confidences": filtered_confidences,
                "average_confidence_by_answer": average_confidence_by_answer,
                "support_count_by_answer": support_count_by_answer,
                "confidence_weight_ratio": confidence_weight_ratio,
                "byzantine_safe": byzantine_safe,
                "fault_tolerance_met": confidence_weight_ratio
                > (0.5 + self.fault_tolerance_threshold / 2.0),
            },
        )

    def normalize_answer(self, answer: str) -> str:
        text = str(answer or "").strip()
        if not text:
            return "0"
        if self.normalization == "none":
            return " ".join(text.split()).lower()
        if self.normalization == "number":
            return self._first_number(text)
        if self._looks_numeric(text):
            return self._first_number(text)
        return " ".join(text.split()).lower()

    def _first_number(self, text: str) -> str:
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if not numbers:
            return "0"
        try:
            value = float(numbers[0])
            return str(int(value)) if value.is_integer() else str(value)
        except ValueError:
            return numbers[0]

    def _looks_numeric(self, text: str) -> bool:
        stripped = text.strip()
        if re.fullmatch(r"(?:answer\s*[:=]\s*)?-?\d+\.?\d*", stripped, re.IGNORECASE):
            return True
        return bool(re.fullmatch(r"[\s\d.,:+\-*/()=]+", stripped))

    def _check_byzantine_safety(
        self,
        *,
        vote_count: int,
        total_nodes: int,
        max_faulty: int,
    ) -> bool:
        if total_nodes <= 0:
            return False
        return vote_count >= max_faulty + 1


def parse_wbft_response(
    agent_id: str,
    content: str,
    *,
    confidence_extraction_method: str = "regex",
    include_unstructured: bool = True,
    fallback_confidence: float = 0.5,
) -> WBFTAgentResponse | None:
    if confidence_extraction_method not in {"json", "regex"}:
        raise ValueError("confidence_extraction_method must be json or regex.")
    payload = _extract_wbft_payload(content)
    if payload is None:
        if confidence_extraction_method == "regex":
            regex_response = _parse_prompt_confidence_response(
                agent_id,
                content,
                fallback_confidence=fallback_confidence,
            )
            if regex_response is not None:
                return regex_response
        if not include_unstructured:
            return None
        return WBFTAgentResponse(
            agent_id=agent_id,
            answer=content.strip(),
            confidence=fallback_confidence,
            raw_content=content,
            metadata={"parser": "unstructured_fallback"},
        )

    return WBFTAgentResponse(
        agent_id=agent_id,
        answer=str(payload.get("answer", "")),
        confidence=float(payload.get("confidence", fallback_confidence)),
        reasoning=str(payload.get("reasoning", "")),
        raw_content=content,
        metadata={
            "parser": "wbft_response_block",
            **{
                str(key): value
                for key, value in payload.items()
                if key not in {"answer", "confidence", "reasoning"}
            },
        },
    )


def _extract_wbft_payload(content: str) -> dict[str, Any] | None:
    if WBFT_RESPONSE_MARKER not in content:
        return None
    block = content.split(WBFT_RESPONSE_MARKER, 1)[1]
    if WBFT_RESPONSE_END in block:
        block = block.split(WBFT_RESPONSE_END, 1)[0]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", block, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else block[block.find("{") : block.rfind("}") + 1]
    if not candidate or not candidate.startswith("{"):
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_prompt_confidence_response(
    agent_id: str,
    content: str,
    *,
    fallback_confidence: float,
) -> WBFTAgentResponse | None:
    confidence = _extract_confidence(content)
    answer = _extract_answer(content)
    if confidence is None and answer is None:
        return None
    return WBFTAgentResponse(
        agent_id=agent_id,
        answer=answer if answer is not None else content.strip(),
        confidence=confidence if confidence is not None else fallback_confidence,
        raw_content=content,
        metadata={
            "parser": "prompt_confidence_regex",
            "confidence_missing": confidence is None,
            "answer_missing": answer is None,
        },
    )


def _extract_confidence(content: str) -> float | None:
    match = re.search(
        r"\bConfidence\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)(%?)",
        content,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2) == "%" or value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def _extract_answer(content: str) -> str | None:
    match = re.search(
        r"\bAnswer\s*[:=]\s*(.+?)(?:\n\s*(?:Confidence|Reasoning)\s*[:=]|\Z)",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    answer = match.group(1).strip()
    if not answer:
        return None
    return answer
