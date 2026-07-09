"""Fisher LDA optimizer for consensus agent quality weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union


HONEST_LABELS = {"honest", "h", "true", "1", "normal"}
BYZANTINE_LABELS = {"byzantine", "malicious", "faulty", "f", "false", "0"}


@dataclass(frozen=True)
class FisherLDASample:
    """Training sample for fitting q = alpha * vc + beta * hc."""

    verified_confidence: float
    historical_confidence: float
    label: str

    def __post_init__(self) -> None:
        _bounded_score("verified_confidence", self.verified_confidence)
        _bounded_score("historical_confidence", self.historical_confidence)
        _normalize_label(self.label)


@dataclass(frozen=True)
class FisherLDAResult:
    """Fitted Fisher LDA quality-function parameters."""

    alpha: float
    beta: float
    theta: float
    fisher_score: float
    honest_mean: Tuple[float, float]
    byzantine_mean: Tuple[float, float]
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "theta": self.theta,
            "fisher_score": self.fisher_score,
            "honest_mean": list(self.honest_mean),
            "byzantine_mean": list(self.byzantine_mean),
            "sample_count": self.sample_count,
        }


RawSample = Union[FisherLDASample, Mapping[str, Any], Sequence[Any]]


def fit_fisher_lda_quality_weights(
    samples: Iterable[RawSample],
    *,
    regularization: float = 1e-6,
) -> FisherLDAResult:
    """
    Fit alpha and beta for q = alpha * vc + beta * hc using Fisher LDA.

    The unconstrained Fisher direction is ``S_w^-1 (mu_H - mu_F)``. Because the
    MARBLE quality function requires interpretable convex weights, the direction
    is oriented toward honest agents and projected onto the 2D probability
    simplex: ``alpha + beta = 1`` and ``alpha,beta >= 0``.
    """
    parsed_samples = [_parse_sample(sample) for sample in samples]
    honest = [
        sample for sample in parsed_samples if _normalize_label(sample.label) == "honest"
    ]
    byzantine = [
        sample
        for sample in parsed_samples
        if _normalize_label(sample.label) == "byzantine"
    ]
    if len(honest) < 1 or len(byzantine) < 1:
        raise ValueError("Fisher LDA requires at least one sample per class.")

    honest_points = [
        (sample.verified_confidence, sample.historical_confidence)
        for sample in honest
    ]
    byzantine_points = [
        (sample.verified_confidence, sample.historical_confidence)
        for sample in byzantine
    ]
    honest_mean = _mean_2d(honest_points)
    byzantine_mean = _mean_2d(byzantine_points)
    delta = (
        honest_mean[0] - byzantine_mean[0],
        honest_mean[1] - byzantine_mean[1],
    )
    scatter = _add_matrix(
        _scatter_matrix(honest_points, honest_mean),
        _scatter_matrix(byzantine_points, byzantine_mean),
    )
    scatter = (
        (scatter[0][0] + regularization, scatter[0][1]),
        (scatter[1][0], scatter[1][1] + regularization),
    )
    direction = _solve_2x2(scatter, delta)
    if _dot(direction, delta) < 0:
        direction = (-direction[0], -direction[1])

    alpha, beta = _project_to_simplex(direction)
    if alpha == 0.5 and beta == 0.5 and abs(delta[0]) + abs(delta[1]) > 0:
        alpha, beta = _project_to_simplex(delta)

    honest_scores = [_quality(point, alpha, beta) for point in honest_points]
    byzantine_scores = [_quality(point, alpha, beta) for point in byzantine_points]
    theta = _best_threshold(honest_scores, byzantine_scores)
    fisher_score = _fisher_score_1d(honest_scores, byzantine_scores)

    return FisherLDAResult(
        alpha=alpha,
        beta=beta,
        theta=theta,
        fisher_score=fisher_score,
        honest_mean=honest_mean,
        byzantine_mean=byzantine_mean,
        sample_count=len(parsed_samples),
    )


def _parse_sample(sample: RawSample) -> FisherLDASample:
    if isinstance(sample, FisherLDASample):
        return sample
    if isinstance(sample, Mapping):
        vc = sample.get("verified_confidence", sample.get("vc"))
        hc = sample.get("historical_confidence", sample.get("hc"))
        label = sample.get("label", sample.get("agent_type"))
        if vc is None or hc is None or label is None:
            raise ValueError("Sample mappings require vc/hc and label fields.")
        return FisherLDASample(float(vc), float(hc), str(label))
    if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)):
        if len(sample) != 3:
            raise ValueError("Sample sequences must be (vc, hc, label).")
        return FisherLDASample(float(sample[0]), float(sample[1]), str(sample[2]))
    raise TypeError(f"Unsupported Fisher LDA sample type: {type(sample)!r}")


def _normalize_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized in HONEST_LABELS:
        return "honest"
    if normalized in BYZANTINE_LABELS:
        return "byzantine"
    raise ValueError(f"Unknown Fisher LDA label: {label!r}")


def _bounded_score(name: str, value: float) -> float:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return float(value)


def _mean_2d(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _scatter_matrix(
    points: Sequence[Tuple[float, float]],
    mean: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    s00 = s01 = s11 = 0.0
    for x0, x1 in points:
        d0 = x0 - mean[0]
        d1 = x1 - mean[1]
        s00 += d0 * d0
        s01 += d0 * d1
        s11 += d1 * d1
    return ((s00, s01), (s01, s11))


def _add_matrix(
    left: Tuple[Tuple[float, float], Tuple[float, float]],
    right: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (
        (left[0][0] + right[0][0], left[0][1] + right[0][1]),
        (left[1][0] + right[1][0], left[1][1] + right[1][1]),
    )


def _solve_2x2(
    matrix: Tuple[Tuple[float, float], Tuple[float, float]],
    vector: Tuple[float, float],
) -> Tuple[float, float]:
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    if abs(determinant) < 1e-12:
        return vector
    return (
        (d * vector[0] - b * vector[1]) / determinant,
        (-c * vector[0] + a * vector[1]) / determinant,
    )


def _project_to_simplex(vector: Tuple[float, float]) -> Tuple[float, float]:
    x, y = vector
    if x <= 0 and y <= 0:
        return (0.5, 0.5)
    x = max(0.0, x)
    y = max(0.0, y)
    total = x + y
    if total <= 0:
        return (0.5, 0.5)
    return (x / total, y / total)


def _quality(point: Tuple[float, float], alpha: float, beta: float) -> float:
    return alpha * point[0] + beta * point[1]


def _best_threshold(
    honest_scores: Sequence[float],
    byzantine_scores: Sequence[float],
) -> float:
    sorted_scores = sorted(set(honest_scores + byzantine_scores))
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    candidates = [
        (left + right) / 2.0
        for left, right in zip(sorted_scores, sorted_scores[1:])
    ]
    candidates = [sorted_scores[0] - 1e-9] + candidates + [sorted_scores[-1] + 1e-9]
    best_theta = candidates[0]
    best_accuracy = -1.0
    for theta in candidates:
        correct_honest = sum(score >= theta for score in honest_scores)
        correct_byzantine = sum(score < theta for score in byzantine_scores)
        accuracy = (correct_honest + correct_byzantine) / (
            len(honest_scores) + len(byzantine_scores)
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_theta = theta
    return max(0.0, min(1.0, best_theta))


def _fisher_score_1d(
    honest_scores: Sequence[float],
    byzantine_scores: Sequence[float],
) -> float:
    honest_mean = sum(honest_scores) / len(honest_scores)
    byzantine_mean = sum(byzantine_scores) / len(byzantine_scores)
    honest_var = sum((score - honest_mean) ** 2 for score in honest_scores)
    byzantine_var = sum((score - byzantine_mean) ** 2 for score in byzantine_scores)
    denominator = honest_var + byzantine_var
    if denominator <= 1e-12:
        return float("inf") if honest_mean != byzantine_mean else 0.0
    return (honest_mean - byzantine_mean) ** 2 / denominator


def _dot(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]
