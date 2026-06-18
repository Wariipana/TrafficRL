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

    Five terms:
      -(wait²)              quadratic wait penalty — grows with congestion
      -0.5*inactive_q       heavy penalty for queue in the RED direction
      -0.1*active_q         mild penalty for queue in the GREEN direction
      +0.1*tp               throughput bonus — auxiliary dense signal
      +0.20*phase_align     per-step signal for serving the heavier direction

    Asymmetry (wrong-phase ≈ −0.70/step, correct-phase ≈ +0.20/step) is intentional:
    it drives the policy away from constant-phase without causing phase-chasing.
    Range: [-1.80, +0.30]
    """
    if not state.intersections:
        return 0.0

    avg_wait = float(np.clip(
        np.mean([s.avg_wait_time for s in state.intersections]) / 600.0, 0.0, 1.0))
    throughput = float(np.clip(
        np.mean([s.throughput for s in state.intersections]) / 50.0, 0.0, 1.0))

    inactive_q_norms: list[float] = []
    active_q_norms:   list[float] = []
    phase_aligns:     list[float] = []
    for s in state.intersections:
        n    = max(1, s.num_lanes)
        half = max(1, n // 2)
        q_ns = float(np.sum(s.queue_length[:half]))
        q_ew = float(np.sum(s.queue_length[half:n]))
        inactive_q = q_ew if s.phase == 0 else q_ns
        active_q   = q_ns if s.phase == 0 else q_ew
        scale = max(50.0 * half, 1.0)
        inactive_q_norms.append(float(np.clip(inactive_q / scale, 0.0, 1.0)))
        active_q_norms.append(float(np.clip(active_q   / scale, 0.0, 1.0)))
        phase_aligns.append((active_q - inactive_q) / (active_q + inactive_q + 1e-3))

    avg_inactive_q  = float(np.mean(inactive_q_norms))
    avg_active_q    = float(np.mean(active_q_norms))
    avg_phase_align = float(np.mean(phase_aligns))

    return -(avg_wait ** 2) - 0.5 * avg_inactive_q - 0.1 * avg_active_q + 0.1 * throughput + 0.20 * avg_phase_align
