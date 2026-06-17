"""
Statistical comparison utilities — no scipy required.
Implements Wilcoxon signed-rank test and Cohen's d for paired / unpaired data.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple


# ---- Wilcoxon signed-rank test (two-sided, exact for n≤25 else normal approx) ----

def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Two-sided Wilcoxon signed-rank test for paired samples (x vs y).
    Returns (W_statistic, p_value).
    Uses normal approximation with continuity correction (valid for n ≥ 5).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diffs = diffs[diffs != 0]   # drop zero differences

    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    abs_d  = np.abs(diffs)
    ranks  = _rank_with_ties(abs_d)
    W_plus  = float(np.sum(ranks[diffs > 0]))
    W_minus = float(np.sum(ranks[diffs < 0]))
    W       = min(W_plus, W_minus)

    # Normal approximation
    mean_W = n * (n + 1) / 4.0
    # Tie correction for variance
    tie_correction = _tie_variance_correction(ranks)
    var_W  = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    if var_W <= 0:
        return W, 1.0

    z      = (W - mean_W + 0.5) / math.sqrt(var_W)   # continuity correction
    p      = 2.0 * _standard_normal_sf(abs(z))        # two-sided
    return W, float(np.clip(p, 0.0, 1.0))


def _rank_with_ties(values: np.ndarray) -> np.ndarray:
    """Average-rank method for tied values."""
    order  = np.argsort(values)
    ranks  = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)

    # Average tied ranks
    sorted_v = values[order]
    i = 0
    while i < len(sorted_v):
        j = i + 1
        while j < len(sorted_v) and sorted_v[j] == sorted_v[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def _tie_variance_correction(ranks: np.ndarray) -> float:
    """Variance correction for ties in Wilcoxon statistic."""
    unique, counts = np.unique(ranks, return_counts=True)
    return float(np.sum(counts * (counts**2 - 1)) / 48.0)


def _standard_normal_sf(z: float) -> float:
    """Survival function of standard normal (upper tail), no scipy needed."""
    return 0.5 * math.erfc(z / math.sqrt(2))


# ---- Mann-Whitney U test (unpaired) ----

def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Two-sided Mann-Whitney U test for independent samples.
    Returns (U_statistic, p_value). Uses normal approximation.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks    = _rank_with_ties(combined)
    U1       = float(np.sum(ranks[:nx])) - nx * (nx + 1) / 2.0
    U2       = nx * ny - U1
    U        = min(U1, U2)

    mean_U = nx * ny / 2.0
    var_U  = nx * ny * (nx + ny + 1) / 12.0
    if var_U <= 0:
        return U, 1.0

    z = (U - mean_U + 0.5) / math.sqrt(var_U)
    p = 2.0 * _standard_normal_sf(abs(z))
    return U, float(np.clip(p, 0.0, 1.0))


# ---- Effect size ----

def cohens_d(x: np.ndarray, y: np.ndarray, paired: bool = False) -> float:
    """
    Cohen's d effect size.
    paired=True: uses std of differences (for matched episodes).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if paired and len(x) == len(y):
        diffs = x - y
        sd    = float(np.std(diffs, ddof=1)) + 1e-12
        return float(np.mean(diffs) / sd)
    pooled = math.sqrt(
        ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
        / (len(x) + len(y) - 2 + 1e-12)
    )
    return float((np.mean(x) - np.mean(y)) / (pooled + 1e-12))


def interpret_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:   return "negligible"
    if ad < 0.5:   return "small"
    if ad < 0.8:   return "medium"
    return "large"


# ---- Pairwise comparison table ----

class PairwiseComparison:
    def __init__(self, name_a: str, name_b: str, metric: str,
                 W: float, p_value: float, d: float, significant: bool):
        self.name_a      = name_a
        self.name_b      = name_b
        self.metric      = metric
        self.W           = W
        self.p_value     = p_value
        self.d           = d
        self.effect_size = interpret_d(d)
        self.significant = significant   # p < 0.05

    def __repr__(self) -> str:
        sig = "*" if self.significant else " "
        return (f"{sig} {self.name_a} vs {self.name_b} [{self.metric}]: "
                f"p={self.p_value:.4f}  d={self.d:+.2f} ({self.effect_size})")


def compare_all_pairs(
    results: dict,   # {name: AlgorithmResult}
    metric: str = "wait",
    alpha: float = 0.05,
) -> list:
    """
    Run all pairwise Wilcoxon tests for a given metric.
    metric: "wait" | "throughput" | "congestion"
    Returns list of PairwiseComparison.
    """
    def get_series(r):
        if metric == "wait":
            return np.array([e.avg_wait_time_s     for e in r.episodes])
        if metric == "throughput":
            return np.array([e.throughput_per_step for e in r.episodes])
        return np.array([e.congestion_spread    for e in r.episodes])

    names = list(results.keys())
    comps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b    = names[i], names[j]
            xa, xb  = get_series(results[a]), get_series(results[b])
            n       = min(len(xa), len(xb))
            W, p    = wilcoxon_signed_rank(xa[:n], xb[:n])
            d       = cohens_d(xa[:n], xb[:n], paired=True)
            comps.append(PairwiseComparison(a, b, metric, W, p, d, p < alpha))
    return comps
