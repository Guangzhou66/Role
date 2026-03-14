"""
Core metric implementations for all experiments.

Every formula from the experiment plan document is implemented here
as a pure function operating on numpy arrays or Python lists.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
#  A1: Basic performance metrics
# ──────────────────────────────────────────────

def accuracy(predictions: List, references: List) -> float:
    """Standard accuracy: fraction of exact matches."""
    assert len(predictions) == len(references)
    if len(predictions) == 0:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)


def exact_match(predictions: List[str], references: List[str]) -> float:
    """Exact match after stripping whitespace."""
    assert len(predictions) == len(references)
    if len(predictions) == 0:
        return 0.0
    return sum(
        p.strip() == r.strip() for p, r in zip(predictions, references)
    ) / len(predictions)


def pass_at_k(n_samples: int, n_correct: int, k: int = 1) -> float:
    """
    Unbiased estimator of pass@k.
    n_samples: total generations, n_correct: # that pass all tests.
    """
    if n_samples - n_correct < k:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= (1.0 - (n_correct) / (n_samples - i))
    return 1.0 - result


# ──────────────────────────────────────────────
#  A1: Cheap–full gap  Δ_t
# ──────────────────────────────────────────────

def cheap_full_gap(perf_full: float, perf_cheap: float) -> float:
    """Δ_t = Perf_full - Perf_cheap"""
    return perf_full - perf_cheap


def normalized_gap(perf_full: float, perf_cheap: float,
                   eps: float = 1e-6) -> float:
    """Normalized gap: (Perf_full - Perf_cheap) / max(Perf_full, ε)"""
    return (perf_full - perf_cheap) / max(perf_full, eps)


# ──────────────────────────────────────────────
#  A2: Budget sensitivity
# ──────────────────────────────────────────────

def degradation_slope(perf_b1: float, perf_b2: float,
                      b1: float, b2: float) -> float:
    """
    S(b1, b2) = (Perf(b1) - Perf(b2)) / (b1 - b2),  b1 > b2
    Measures per-unit-budget performance loss.
    """
    assert b1 > b2, "b1 must be greater than b2"
    return (perf_b1 - perf_b2) / (b1 - b2)


def aubc(budgets: List[float], performances: List[float]) -> float:
    """
    Area Under the Budget-performance Curve (trapezoidal rule).
    AUBC = ∫ Perf(b) db ≈ Σ (Perf(b_i) + Perf(b_{i+1})) / 2 * (b_{i+1} - b_i)
    """
    assert len(budgets) == len(performances)
    pairs = sorted(zip(budgets, performances))
    area = 0.0
    for i in range(len(pairs) - 1):
        b_i, p_i = pairs[i]
        b_next, p_next = pairs[i + 1]
        area += (p_i + p_next) / 2.0 * (b_next - b_i)
    return area


# ──────────────────────────────────────────────
#  A3: Difficulty-stratified analysis
# ──────────────────────────────────────────────

def compute_difficulty_scores(
    correct_per_seed: List[List[bool]],
) -> np.ndarray:
    """
    Compute difficulty d(x) = 1 - (avg correct rate over M seeds).
    correct_per_seed: list of M boolean-lists, each of length N.
    Returns: array of shape (N,) with values in [0, 1].
    """
    M = len(correct_per_seed)
    N = len(correct_per_seed[0])
    correct_counts = np.zeros(N)
    for correct_list in correct_per_seed:
        for i, c in enumerate(correct_list):
            if c:
                correct_counts[i] += 1
    return 1.0 - correct_counts / M


def stratify_by_difficulty(
    difficulty_scores: np.ndarray, n_bins: int = 3
) -> Dict[str, np.ndarray]:
    """
    Split indices into difficulty groups by quantile.
    Returns dict mapping group name -> array of sample indices.
    """
    thresholds = np.quantile(difficulty_scores,
                             np.linspace(0, 1, n_bins + 1))
    labels = ["easy", "medium", "hard"] if n_bins == 3 else \
             [f"group_{i}" for i in range(n_bins)]

    groups = {}
    for i, label in enumerate(labels):
        lo = thresholds[i]
        hi = thresholds[i + 1]
        if i == len(labels) - 1:
            mask = (difficulty_scores >= lo) & (difficulty_scores <= hi)
        else:
            mask = (difficulty_scores >= lo) & (difficulty_scores < hi)
        groups[label] = np.where(mask)[0]
    return groups


def fail_rate_by_group(
    predictions: List, references: List,
    group_indices: np.ndarray
) -> float:
    """FailRate_d = fraction of errors in a difficulty group."""
    if len(group_indices) == 0:
        return 0.0
    errors = sum(
        predictions[i] != references[i] for i in group_indices
    )
    return errors / len(group_indices)


def failure_concentration_ratio(
    predictions: List, references: List,
    hard_indices: np.ndarray
) -> float:
    """
    FCR_hard = (# errors in hard group) / (total # errors)
    """
    total_errors = sum(p != r for p, r in zip(predictions, references))
    if total_errors == 0:
        return 0.0
    hard_errors = sum(
        predictions[i] != references[i] for i in hard_indices
    )
    return hard_errors / total_errors


def hard_vs_easy_amplification(
    fail_rate_hard: float, fail_rate_easy: float,
    eps: float = 1e-6
) -> float:
    """HFA = FailRate_hard / max(FailRate_easy, ε)"""
    return fail_rate_hard / max(fail_rate_easy, eps)


# ──────────────────────────────────────────────
#  B7 / C1: Recovery & method alignment
# ──────────────────────────────────────────────

def recovery_rate(
    predictions_method: List, predictions_cheap: List,
    predictions_full: List, references: List
) -> float:
    """
    RecRate(m) = fraction of baseline-failure samples recovered by method m.
    Failure set F = {x | full correct, cheap wrong}.
    """
    failure_set = [
        i for i in range(len(references))
        if predictions_full[i] == references[i]
        and predictions_cheap[i] != references[i]
    ]
    if len(failure_set) == 0:
        return 0.0
    recovered = sum(
        predictions_method[i] == references[i] for i in failure_set
    )
    return recovered / len(failure_set)


def adaptation_recovery_ratio(
    perf_adapt: float, perf_compressed: float,
    perf_full: float, eps: float = 1e-6
) -> float:
    """
    R^adapt = (P_adapt - P_cmp) / (P_full - P_cmp + ε)
    """
    return (perf_adapt - perf_compressed) / (perf_full - perf_compressed + eps)


def teacher_consistency(
    predictions_compressed: List, predictions_full: List
) -> float:
    """C^teacher = fraction where compressed output == full output."""
    n = len(predictions_compressed)
    if n == 0:
        return 0.0
    return sum(
        c == f for c, f in zip(predictions_compressed, predictions_full)
    ) / n


# ──────────────────────────────────────────────
#  C1: Method-to-problem alignment (gains & synergy)
# ──────────────────────────────────────────────

def selection_gain(perf_role: float, perf_rec: float) -> float:
    """G_sel = P^role - P^rec"""
    return perf_role - perf_rec


def adaptation_gain(perf_rec_adp: float, perf_rec: float) -> float:
    """G_adp = P^{rec+adp} - P^rec"""
    return perf_rec_adp - perf_rec


def joint_gain(perf_role_adp: float, perf_rec: float) -> float:
    """G_joint = P^{role+adp} - P^rec"""
    return perf_role_adp - perf_rec


def synergy(perf_role_adp: float, perf_role: float,
            perf_rec_adp: float, perf_rec: float) -> float:
    """S = P^{role+adp} - P^role - P^{rec+adp} + P^rec"""
    return perf_role_adp - perf_role - perf_rec_adp + perf_rec


# ──────────────────────────────────────────────
#  C5: Oracle gap
# ──────────────────────────────────────────────

def oracle_gap(perf_oracle: float, perf_method: float) -> float:
    """G^oracle = P^oracle - P^method"""
    return perf_oracle - perf_method


# ──────────────────────────────────────────────
#  C6: Statistical reliability
# ──────────────────────────────────────────────

def mean_std_ci(values: List[float], confidence: float = 0.95
                ) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute mean, std, and confidence interval from multiple seeds.
    Returns (mean, std, (ci_lower, ci_upper)).
    """
    arr = np.array(values)
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    n = len(arr)

    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    margin = z * sigma / np.sqrt(n)
    return float(mu), float(sigma), (float(mu - margin), float(mu + margin))


# ──────────────────────────────────────────────
#  C3: Efficiency-aware score
# ──────────────────────────────────────────────

def efficiency_aware_score(perf: float, cost: float,
                           lam: float = 0.1) -> float:
    """EAS(m, b) = Perf(m, b) - λ · Cost(m, b)"""
    return perf - lam * cost
