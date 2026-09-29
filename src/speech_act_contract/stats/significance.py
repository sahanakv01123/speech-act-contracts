from __future__ import annotations

from math import comb, sqrt


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a Bernoulli proportion."""
    if total <= 0:
        raise ValueError("total must be positive")

    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    half = (z * sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * total)) / total)) / denom
    return center - half, center + half


def exact_sign_test_pvalue(positive: int, negative: int) -> float:
    """Two-sided exact sign test under the null p=0.5."""
    if positive < 0 or negative < 0:
        raise ValueError("counts must be non-negative")

    total = positive + negative
    if total == 0:
        return 1.0

    probs = [comb(total, k) * (0.5**total) for k in range(total + 1)]
    observed = probs[min(positive, negative)]
    return min(1.0, sum(prob for prob in probs if prob <= observed + 1e-15))


def paired_binary_improvement_pvalue(improved: int, worsened: int) -> float:
    """McNemar-equivalent exact sign test for paired binary changes."""
    return exact_sign_test_pvalue(improved, worsened)
