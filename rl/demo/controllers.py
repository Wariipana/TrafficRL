"""
Demo heuristic controllers for presentation mode (--demo flag in run.sh).

Each controller mimics the expected behaviour of a RL algorithm family
without actually running a trained model — useful for live demos when
training is still in progress.  All controllers receive the same obs_d
dict the real agents would see and return a phase array (int64, shape (N,)).

Progression:
  DemoBaseline  — broken fixed-time cycles (85-90 % phase 0)
  DemoPPO       — centralized timer, 50/50 split, all lights in sync
  DemoIPPO      — per-intersection reactive: responds to local queue imbalance
  DemoHRL       — zone-coordinated: neighbour pressure adjusts thresholds + offsets
"""
from __future__ import annotations

import numpy as np


# ── 1. Baseline ───────────────────────────────────────────────────────────────

class DemoBaseline:
    """
    Badly-configured fixed-time baseline.

    Per-light random period (800-1200 sim-steps) with only 10-15 % of each
    cycle giving NS traffic a green phase.  Unsynchronised offsets mean there
    is no accidental green-wave either.  Identical logic to FixedRandomRunner
    so metrics match the benchmark baseline.
    """
    name = "Semáforos mal configurados (baseline)"

    def __init__(self, n_lights: int, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.periods   = rng.integers(800, 1201, size=n_lights)
        self.green_dur = np.maximum(
            100,
            (self.periods * rng.uniform(0.10, 0.15, size=n_lights)).astype(np.int64),
        )
        self.offsets = np.array(
            [rng.integers(0, p) for p in self.periods], dtype=np.int64
        )

    def get_phases(self, obs_d: dict, step: int, n: int) -> np.ndarray:
        pos = (step + self.offsets) % self.periods
        return (pos >= (self.periods - self.green_dur)).astype(np.int64)


# ── 2. PPO centralizado ───────────────────────────────────────────────────────

class DemoPPO:
    """
    Centralized fixed-cycle with equal split.

    Represents a naive centralised policy that has learned "switch every N
    steps" regardless of actual traffic load.  Better than the baseline
    (50/50 instead of 85/15) but still ignores per-intersection state, so
    congestion builds when traffic is asymmetric.
    """
    name = "PPO centralizado"

    def __init__(self, n_lights: int, half_period: int = 150) -> None:
        self.n           = n_lights
        self.half_period = half_period  # steps per phase (both phases equal)

    def get_phases(self, obs_d: dict, step: int, n: int) -> np.ndarray:
        phase = (step // self.half_period) % 2
        return np.full(n, phase, dtype=np.int64)


# ── 3. IPPO + GNN ─────────────────────────────────────────────────────────────

class DemoIPPO:
    """
    Per-intersection reactive controller.

    Each light independently looks at its own NS vs EW queue imbalance and
    switches to serve the heavier direction once the minimum green time has
    elapsed.  Mimics what a trained IPPO agent with sufficient phase_align
    signal should learn to do.
    """
    name = "IPPO + GNN"

    MIN_GREEN_S = 12.0   # seconds; mirrors motor's DEFAULT_MIN_GREEN
    THRESHOLD   = 0.25   # |imbalance| above which we consider switching

    def get_phases(self, obs_d: dict, step: int, n: int) -> np.ndarray:
        phases = np.zeros(n, dtype=np.int64)
        for i in range(n):
            obs = obs_d.get(f"light_{i}")
            if obs is None:
                continue
            ql   = obs["queue_length"]
            half = len(ql) // 2
            q_ns = float(np.sum(ql[:half]))
            q_ew = float(np.sum(ql[half:]))
            cur   = int(obs["current_phase"])
            timer = float(obs["phase_timer"][0])   # seconds active

            if timer < self.MIN_GREEN_S:
                phases[i] = cur
                continue

            imbalance = (q_ns - q_ew) / (q_ns + q_ew + 1e-3)
            if imbalance > self.THRESHOLD and cur == 1:
                phases[i] = 0   # NS heavier → switch to NS_GREEN
            elif imbalance < -self.THRESHOLD and cur == 0:
                phases[i] = 1   # EW heavier → switch to EW_GREEN
            else:
                phases[i] = cur
        return phases


# ── 4. HRL jerárquico ─────────────────────────────────────────────────────────

class DemoHRL:
    """
    Zone-coordinated heuristic.

    Two improvements over DemoIPPO:
      1. Neighbour pressure lowers the switching threshold, so a light is
         more aggressive when its neighbours are also congested.
      2. A small per-light phase offset staggers switches across the grid,
         creating a rudimentary green-wave effect that reduces stop-and-go.

    Mimics the expected emergent behaviour of a trained HRL Manager+Worker
    where the Manager sets zone-level targets that bias local decisions.
    """
    name = "HRL jerárquico"

    MIN_GREEN_S     = 12.0
    BASE_THRESHOLD  = 0.20
    NB_WEIGHT       = 0.40   # how much neighbour congestion lowers the threshold
    WAVE_OFFSET_S   = 8.0    # seconds between adjacent zones

    def __init__(self, n_lights: int, seed: int = 3) -> None:
        rng = np.random.default_rng(seed)
        # Each light gets a small offset so green phases ripple across the grid.
        self.offsets = (rng.integers(0, n_lights, size=n_lights).astype(float)
                        * self.WAVE_OFFSET_S)

    def get_phases(self, obs_d: dict, step: int, n: int) -> np.ndarray:
        phases = np.zeros(n, dtype=np.int64)
        for i in range(n):
            obs = obs_d.get(f"light_{i}")
            if obs is None:
                continue
            ql   = obs["queue_length"]
            half = len(ql) // 2
            q_ns = float(np.sum(ql[:half]))
            q_ew = float(np.sum(ql[half:]))
            cur   = int(obs["current_phase"])
            timer = float(obs["phase_timer"][0])

            # Effective timer includes the wave offset so adjacent lights
            # don't switch simultaneously.
            effective_timer = timer - self.offsets[i]
            if effective_timer < self.MIN_GREEN_S:
                phases[i] = cur
                continue

            nb_queue   = float(obs["neighbor_summary"][0])   # normalised [0,1]
            threshold  = self.BASE_THRESHOLD * (1.0 - self.NB_WEIGHT * nb_queue)
            threshold  = max(0.05, threshold)

            imbalance = (q_ns - q_ew) / (q_ns + q_ew + 1e-3)
            if imbalance > threshold and cur == 1:
                phases[i] = 0
            elif imbalance < -threshold and cur == 0:
                phases[i] = 1
            else:
                phases[i] = cur
        return phases


# ── factory ───────────────────────────────────────────────────────────────────

DEMO_ALGO_LABELS: dict[str, str] = {
    "demo_baseline": DemoBaseline.name,
    "demo_ppo":      DemoPPO.name,
    "demo_ippo":     DemoIPPO.name,
    "demo_hrl":      DemoHRL.name,
}


def make_controller(algo: str, n_lights: int):
    """Return an instantiated controller for the given demo algo key."""
    if algo == "demo_baseline":
        return DemoBaseline(n_lights)
    if algo == "demo_ppo":
        return DemoPPO(n_lights)
    if algo == "demo_ippo":
        return DemoIPPO()
    if algo == "demo_hrl":
        return DemoHRL(n_lights)
    raise ValueError(f"Unknown demo algo: {algo!r}")
