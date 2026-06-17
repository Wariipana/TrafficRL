from __future__ import annotations
import numpy as np
import gymnasium
from .data_types import EnvConfig, GraphData, MAX_LANES

MAX_VEH_PER_LANE  = 50.0
MAX_SPEED_MS      = 50.0   # ~180 km/h
MAX_WAIT_S        = 600.0  # 10 minutes
MAX_PHASE_TIME_S  = 120.0


def build_observation_space(graph: GraphData, cfg: EnvConfig) -> gymnasium.spaces.Dict:
    """
    Build a Gymnasium Dict observation space for the centralized agent.
    Shapes: (num_lights, MAX_LANES) for per-lane fields, (num_lights,) for scalar fields.
    """
    n = graph.num_lights

    return gymnasium.spaces.Dict({
        "vehicles_per_lane": gymnasium.spaces.Box(
            low=0.0, high=MAX_VEH_PER_LANE,
            shape=(n, MAX_LANES), dtype=np.float32,
        ),
        "queue_length": gymnasium.spaces.Box(
            low=0.0, high=MAX_VEH_PER_LANE,
            shape=(n, MAX_LANES), dtype=np.float32,
        ),
        "avg_speed": gymnasium.spaces.Box(
            low=0.0, high=MAX_SPEED_MS,
            shape=(n, MAX_LANES), dtype=np.float32,
        ),
        "avg_wait_time": gymnasium.spaces.Box(
            low=0.0, high=MAX_WAIT_S,
            shape=(n,), dtype=np.float32,
        ),
        "current_phase": gymnasium.spaces.MultiDiscrete([2] * n),
        "phase_timer": gymnasium.spaces.Box(
            low=0.0, high=MAX_PHASE_TIME_S,
            shape=(n,), dtype=np.float32,
        ),
    })


def build_action_space(graph: GraphData) -> gymnasium.spaces.MultiDiscrete:
    """
    Centralized agent: one discrete action per traffic light.
    action[i] in {0=NS_GREEN, 1=EW_GREEN}.
    """
    return gymnasium.spaces.MultiDiscrete([2] * graph.num_lights)
