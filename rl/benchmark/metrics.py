"""
Benchmark metrics for TrafficRL — decoupled from the reward signal.

EpisodeMetrics:   single-episode results
AlgorithmResult:  N episodes for one algorithm
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class EpisodeMetrics:
    # Throughput
    total_vehicles_served: float   # cumulative vehicles that completed their route
    throughput_per_step:   float   # avg vehicles / step

    # Delay
    avg_wait_time_s:  float        # mean over all intersections, final step
    max_wait_time_s:  float        # worst single vehicle wait observed in episode

    # Flow quality
    congestion_spread: float       # 0-1, fraction of intersections congested
    avg_speed_ms:      float       # mean speed across all intersection lanes (m/s)

    # Efficiency
    steps_completed: int
    episode_id:      int


@dataclass
class AlgorithmResult:
    name:           str
    config_label:   str            # e.g. "city_small", "city_medium"
    episodes:       List[EpisodeMetrics] = field(default_factory=list)

    # Derived — populated by compute_summary()
    mean_wait:        float = 0.0
    std_wait:         float = 0.0
    ci95_wait:        tuple[float, float] = (0.0, 0.0)
    mean_throughput:  float = 0.0
    std_throughput:   float = 0.0
    ci95_throughput:  tuple[float, float] = (0.0, 0.0)
    mean_congestion:  float = 0.0
    std_congestion:   float = 0.0
    mean_speed:       float = 0.0

    def compute_summary(self, bootstrap_n: int = 2000, rng_seed: int = 0) -> None:
        """Compute means, std, and 95% bootstrap CIs from episode list."""
        if not self.episodes:
            return
        waits = np.array([e.avg_wait_time_s  for e in self.episodes])
        tps   = np.array([e.throughput_per_step for e in self.episodes])
        congs = np.array([e.congestion_spread   for e in self.episodes])
        spds  = np.array([e.avg_speed_ms        for e in self.episodes])

        self.mean_wait       = float(np.mean(waits))
        self.std_wait        = float(np.std(waits, ddof=1) if len(waits) > 1 else 0.0)
        self.mean_throughput = float(np.mean(tps))
        self.std_throughput  = float(np.std(tps, ddof=1)   if len(tps) > 1 else 0.0)
        self.mean_congestion = float(np.mean(congs))
        self.mean_speed      = float(np.mean(spds))

        self.ci95_wait       = _bootstrap_ci(waits, bootstrap_n, rng_seed)
        self.ci95_throughput = _bootstrap_ci(tps,   bootstrap_n, rng_seed)


def _bootstrap_ci(
    data: np.ndarray,
    n_samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Non-parametric bootstrap 95% CI for the mean."""
    if len(data) < 2:
        v = float(data[0]) if len(data) == 1 else 0.0
        return (v, v)
    rng      = np.random.default_rng(seed)
    boot     = rng.choice(data, size=(n_samples, len(data)), replace=True)
    means    = boot.mean(axis=1)
    lo, hi   = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
