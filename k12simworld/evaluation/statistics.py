"""Small, dependency-free statistical routines used by the paper tables."""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import List, Optional, Sequence, Tuple


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            result[index] = average_rank
        position = end
    return result


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires equal sequences of length >= 2")
    x, y = _ranks(left), _ranks(right)
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else float("nan")


def krippendorff_alpha(rows: Sequence[Sequence[Optional[float]]]) -> float:
    """Ordinal alpha using squared-distance disagreement and missing-value support."""
    observed: List[float] = []
    pooled: List[float] = []
    for row in rows:
        values = [float(value) for value in row if value is not None]
        pooled.extend(values)
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                observed.append((values[left] - values[right]) ** 2)
    if not observed or len(pooled) < 2:
        return float("nan")
    expected = [
        (pooled[left] - pooled[right]) ** 2
        for left in range(len(pooled))
        for right in range(left + 1, len(pooled))
    ]
    de = mean(expected)
    return 1.0 - mean(observed) / de if de else 1.0


def bootstrap_paired_difference(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> Tuple[float, float, float]:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired bootstrap requires non-empty equal-length inputs")
    differences = [right - left for left, right in zip(baseline, treatment)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean(differences[rng.randrange(len(differences))] for _ in differences))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low = estimates[max(0, int(alpha * samples))]
    high = estimates[min(samples - 1, int((1.0 - alpha) * samples) - 1)]
    return mean(differences), low, high


def paired_randomisation_pvalue(
    baseline: Sequence[float], treatment: Sequence[float], samples: int = 20_000, seed: int = 2026
) -> float:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired randomisation requires non-empty equal-length inputs")
    differences = [right - left for left, right in zip(baseline, treatment)]
    observed = abs(mean(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        candidate = abs(mean(value if rng.random() < 0.5 else -value for value in differences))
        extreme += candidate >= observed
    return (extreme + 1) / (samples + 1)


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted
