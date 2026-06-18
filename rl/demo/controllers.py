"""
Presentation-mode heuristic controllers (run.sh --demo).

Each controller mimics the *expected* behaviour of an RL algorithm family
without trained weights.  The C++ sim enforces min_green=10 s on its own,
so Python controllers simply declare the *desired* phase every step — the
engine handles the actual minimum-green constraint.

Progression:
  PresentationBaseline  — badly-configured fixed-time cycles (80 % NS-biased)
  PresentationPPO       — centralised 50/50 cycle; all lights in sync
  PresentationIPPO      — per-intersection reactive: always serves the heavier queue
  PresentationHRL       — zone-coordinated wave: groups of lights stagger switches
"""
from __future__ import annotations

import numpy as np


# ── 1. Baseline ───────────────────────────────────────────────────────────────

class PresentationBaseline:
    """
    Badly-configured fixed-time baseline.

    Each light has a random period (40-60 sim-seconds) with only ~17 % of the
    cycle giving EW traffic a green phase (the rest NS).  Unsynchronised offsets
    mean there is no accidental green-wave.  Creates heavy EW congestion that is
    visually obvious in the webviz.
    """
    name = "Semáforos mal configurados (baseline)"

    # C++ min_green = 10 s = 100 steps at dt=0.1.  green_dur must be >= 100
    # steps to let EW traffic actually move when it eventually gets its slot.
    _MIN_STEPS = 100

    def __init__(self, n_lights: int, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        # period: 400-600 sim-steps (40-60 seconds at dt=0.1)
        self.periods = rng.integers(400, 601, size=n_lights)
        # green_dur: 15-20 % of period, but at least min_green so C++ honours it
        raw = (self.periods * rng.uniform(0.15, 0.20, size=n_lights)).astype(np.int64)
        self.green_dur = np.maximum(self._MIN_STEPS, raw)
        self.offsets = np.array(
            [rng.integers(0, p) for p in self.periods], dtype=np.int64
        )

    def get_phases(self, obs_d: dict, step: int, n: int) -> np.ndarray:
        pos = (step + self.offsets) % self.periods
        # phase=1 (EW_GREEN) only during the last green_dur steps of each cycle;
        # the remaining 80-85 % is phase=0 (NS_GREEN)
        return (pos >= (self.periods - self.green_dur)).astype(np.int64)


# ── 2. PPO centralizado ───────────────────────────────────────────────────────

class PresentationPPO:
    """
    Centralised fixed-cycle with equal 50/50 split.

    All lights flip simultaneously every HALF_PERIOD steps — like a naive
    centralised policy that learned "switch globally at a fixed rate" but
    ignores per-intersection load.  The half-period matches C++ min_green
    (100 steps = 10 s) so each switch is always honoured.  Visible drawback:
    every intersection switches at the same instant, creating traffic waves
    that crash into the next red light.
    """
    name = "PPO centralizado"

    def __init__(self, n_lights: int, half_period: int = 100) -> None:
        self.n           = n_lights
        self.half_period = half_period   # 100 steps = 10 s = C++ min_green

    def get_phases(self, obs_d: dict, step: int, n: int) -> np.ndarray:
        phase = (step // self.half_period) % 2
        return np.full(n, phase, dtype=np.int64)


# ── 3. IPPO + GNN ─────────────────────────────────────────────────────────────

class PresentationIPPO:
    """
    Per-intersection reactive controller.

    Each light independently requests whichever phase serves the heavier queue
    direction (NS vs EW).  The C++ engine enforces the 10-second minimum green,
    so the Python controller simply expresses the *desired* phase every step.
    This causes each light to switch as soon as its minimum green expires when
    the other direction has more queued vehicles.

    Visibly better than baseline/PPO: queues stay balanced per intersection,
    no global synchronisation artefacts.
    """
    name = "IPPO + GNN"

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
            # Always request the phase that serves the heavier direction.
            # C++ will delay the actual switch until min_green (10 s) has elapsed.
            phases[i] = 0 if q_ns >= q_ew else 1
        return phases


# ── 4. HRL jerárquico ─────────────────────────────────────────────────────────

class PresentationHRL:
    """
    Zone-coordinated wave controller.

    Lights are divided into four zones.  Within each zone, all lights switch
    simultaneously; zones are staggered by ZONE_OFFSET steps so a "green wave"
    ripples across the grid every CYCLE_STEPS.

    When local queue imbalance is large (> OVERRIDE_THRESH), the light ignores
    the wave and serves its heavier direction — mimicking the HRL Worker
    overriding a Manager zone-target when congestion is severe.

    Visibly better than IPPO: coordinated switches create green waves along
    corridors, reducing stop-and-go.
    """
    name = "HRL jerárquico"

    CYCLE_STEPS    = 300   # steps for one full zone-wave cycle (30 s at dt=0.1)
    ZONE_OFFSET    = 75    # stagger between adjacent zones (7.5 s)
    OVERRIDE_THRESH = 0.30  # imbalance above which local queue overrides the wave

    def __init__(self, n_lights: int, seed: int = 3) -> None:
        # Assign each light a zone (0-3).  With a 4×4 grid and row-major
        # numbering, lights 0-3 → zone 0, 4-7 → zone 1, etc.
        self.zone_offset = np.array(
            [(i // max(1, n_lights // 4)) % 4 * self.ZONE_OFFSET
             for i in range(n_lights)],
            dtype=np.int64,
        )

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

            # Local queue preference
            imbalance = (q_ns - q_ew) / (q_ns + q_ew + 1e-3)

            if abs(imbalance) > self.OVERRIDE_THRESH:
                # Strong local imbalance → Worker overrides Manager wave
                phases[i] = 0 if imbalance > 0 else 1
            else:
                # Follow the zone wave (green wave propagation)
                wave_pos = (step + self.zone_offset[i]) % self.CYCLE_STEPS
                phases[i] = int(wave_pos >= self.CYCLE_STEPS // 2)
        return phases


# ── factory ───────────────────────────────────────────────────────────────────

DEMO_ALGO_LABELS: dict[str, str] = {
    "demo_baseline": PresentationBaseline.name,
    "demo_ppo":      PresentationPPO.name,
    "demo_ippo":     PresentationIPPO.name,
    "demo_hrl":      PresentationHRL.name,
}


def make_controller(algo: str, n_lights: int):
    """Return an instantiated controller for the given internal algo key."""
    if algo == "demo_baseline":
        return PresentationBaseline(n_lights)
    if algo == "demo_ppo":
        return PresentationPPO(n_lights)
    if algo == "demo_ippo":
        return PresentationIPPO()
    if algo == "demo_hrl":
        return PresentationHRL(n_lights)
    raise ValueError(f"Unknown presentation algo: {algo!r}")
