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
    Shared reward signal used by both the centralized PPO env and (as a
    global component) the MARL env.

    All per-intersection metrics are normalized to [0, 1] before weighting
    so that each term contributes on a comparable scale and the total reward
    stays roughly in [-1.2, +0.8].  Raw values varied by 2–3 orders of
    magnitude (wait in seconds, queue in vehicles) which made the value
    function converge slowly.
    """
    if not state.intersections:
        return 0.0

    n = len(state.intersections)

    # ---- local component (per-intersection, averaged) ----
    avg_wait   = np.mean([s.avg_wait_time         for s in state.intersections]) / 600.0
    stopped    = np.mean([np.mean(s.queue_length) for s in state.intersections]) / 50.0
    max_queue  = np.mean([np.max(s.queue_length)  for s in state.intersections]) / 50.0
    throughput = np.mean([s.throughput            for s in state.intersections]) / 50.0

    local_r = (
        - cfg.alpha * float(np.clip(avg_wait,   0.0, 1.0))
        - cfg.beta  * float(np.clip(stopped,    0.0, 1.0))
        - cfg.gamma * float(np.clip(max_queue,  0.0, 1.0))
        + cfg.delta * float(np.clip(throughput, 0.0, 1.0))
    )

    # ---- phase-load pressure (averaged across intersections) ----
    # Reward for actively serving the more congested direction.
    # Without this signal the policy converges to a near-constant phase bias
    # and keeps one direction red for the full max_green window.
    pressure_r = cfg.pressure * float(
        np.mean([_phase_match(s) for s in state.intersections])
    )

    # ---- global component ----
    global_tp_norm = float(np.clip(state.total_throughput / max(n * 50.0, 1.0), 0.0, 1.0))
    global_r = (
        + cfg.eta  * global_tp_norm
        - cfg.zeta * float(state.congestion_spread)
    )

    return float(cfg.local_weight * (local_r + pressure_r) + cfg.global_weight * global_r)
