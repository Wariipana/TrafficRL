from __future__ import annotations
from .data_types import StateSnapshot, RewardConfig
import numpy as np


def _phase_match(s) -> float:
    """
    Compute how well the current phase matches the actual traffic load.

    Splits queue_length into two halves: the C++ motor stores NS-approach
    lanes before EW-approach lanes, so queue[:half] ≈ phase-0 load and
    queue[half:] ≈ phase-1 load.

    Returns a value in [-1, +1]:
      +1  the active phase is serving the fully loaded direction
      -1  the active phase is wasting green on the empty direction
       0  both directions have equal load
    """
    n    = max(1, s.num_lanes)
    half = max(1, n // 2)
    q_ns = float(np.sum(s.queue_length[:half]))
    q_ew = float(np.sum(s.queue_length[half:n]))
    q_active   = q_ns if s.phase == 0 else q_ew
    q_inactive = q_ew if s.phase == 0 else q_ns
    total_q    = q_active + q_inactive + 1e-3
    return (q_active - q_inactive) / total_q


def compute_reward(state: StateSnapshot, cfg: RewardConfig) -> float:
    """
    Reward signal for the centralized PPO env.

    Three terms, all in [0, 1]:
      -(wait²)      quadratic wait penalty — gradient grows with congestion,
                    preventing the policy from settling at a mediocre plateau
      -0.4*queue    linear queue penalty — responds immediately to phase
                    changes (fast credit assignment, fixes frozen gradients)
      +0.1*tp       small throughput bonus — auxiliary dense signal

    Range: [-1.4, +0.1]
    """
    if not state.intersections:
        return 0.0

    avg_wait  = float(np.clip(
        np.mean([s.avg_wait_time for s in state.intersections]) / 600.0, 0.0, 1.0))
    avg_queue = float(np.clip(
        np.mean([np.mean(s.queue_length[:max(1, s.num_lanes)])
                 for s in state.intersections]) / 50.0, 0.0, 1.0))
    throughput = float(np.clip(
        np.mean([s.throughput for s in state.intersections]) / 50.0, 0.0, 1.0))

    return -(avg_wait ** 2) - 0.4 * avg_queue + 0.1 * throughput
