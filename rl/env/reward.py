from __future__ import annotations
from .data_types import StateSnapshot, RewardConfig
import numpy as np


def compute_reward(state: StateSnapshot, cfg: RewardConfig) -> float:
    """
    Two-level reward: local (per intersection) + global (zone-level).
    Separated from evaluation metrics to avoid Goodhart's Law.
    """
    if not state.intersections:
        return 0.0

    # Local component: aggregate over all intersections
    avg_wait   = np.mean([s.avg_wait_time for s in state.intersections])
    stopped    = np.mean([np.sum(s.queue_length) for s in state.intersections])
    max_queue  = np.mean([np.max(s.queue_length) for s in state.intersections])
    throughput = float(np.mean([s.throughput for s in state.intersections]))

    local_r = (
        - cfg.alpha * avg_wait
        - cfg.beta  * stopped
        - cfg.gamma * max_queue
        + cfg.delta * throughput
    )

    # Global component: zone-level metrics from the state snapshot
    global_r = (
        + cfg.eta  * state.total_throughput
        - cfg.zeta * state.congestion_spread
    )

    return float(cfg.local_weight * local_r + cfg.global_weight * global_r)
