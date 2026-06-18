from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

MAX_LANES  = 8
MAX_LIGHTS = 256
MAX_VEHICLES_EXPORT = 4096   # mirrors MAX_VEHICLES in C++ types.hpp

# ---- Graph topology (read once per reset) ----

@dataclass
class NodeRecord:
    id: int
    x: float
    y: float
    zone: int
    has_light: bool
    light_id: int
    num_outgoing: int


@dataclass
class EdgeRecord:
    id: int
    from_node: int
    to_node: int
    length: float
    num_lanes: int
    speed_limit: float
    direction: int


@dataclass
class GraphData:
    nodes: list[NodeRecord]
    edges: list[EdgeRecord]
    light_node_ids: list[int]   # node ids that have traffic lights
    num_lights: int

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


# ---- Per-step state snapshot ----

@dataclass
class IntersectionSnapshot:
    id: int
    phase: int
    num_lanes: int
    phase_timer_s: float
    vehicles_per_lane: np.ndarray   # shape (MAX_LANES,)
    queue_length: np.ndarray        # shape (MAX_LANES,)
    avg_speed: np.ndarray           # shape (MAX_LANES,)
    avg_wait_time: float
    throughput: float
    in_all_red: bool = False        # inter-phase transition (amber render); default keeps
                                    # existing constructors working


@dataclass
class StateSnapshot:
    sim_tick: int
    num_intersections: int
    num_vehicles: int
    sim_time_s: float
    episode_step: int
    terminated: bool
    truncated: bool
    intersections: list[IntersectionSnapshot]
    total_throughput: float
    avg_wait_global: float
    max_wait_global: float
    congestion_spread: float


# ---- Environment configuration ----

@dataclass
class CityConfig:
    grid_width: int = 4
    grid_height: int = 4
    block_size: float = 100.0
    avenue_probability: float = 0.3
    street_probability: float = 0.7
    residential_ratio: float = 0.6
    commercial_ratio: float = 0.3
    industrial_ratio: float = 0.1
    traffic_light_density: float = 0.8
    seed: int = 42


@dataclass
class RewardConfig:
    alpha:        float = 0.4   # wait penalty            (input normalized to [0,1])
    beta:         float = 0.3   # queue density penalty   (input normalized to [0,1])
    gamma:        float = 0.2   # max-queue penalty       (input normalized to [0,1])
    delta:        float = 0.2   # throughput reward       (input normalized to [0,1])
    eta:          float = 0.3   # global throughput reward
    zeta:         float = 0.2   # congestion spread penalty
    pressure:     float = 0.5   # phase-load matching signal  (output in [-1, 1])
    local_weight: float = 0.7
    global_weight: float = 0.3


@dataclass
class EnvConfig:
    city: CityConfig = field(default_factory=CityConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation_mode: str = "perfect"        # "perfect" | "sensor_noise" | "camera_vision"
    communication_mode: str = "none"         # "none" | "neighbors" | "global"
    episode_length_steps: int = 2000
    dt: float = 0.1
    shm_prefix: str = "trafficrl"

    @classmethod
    def from_yaml(cls, path: str) -> "EnvConfig":
        import os
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        city_raw   = raw.get("city", {})
        reward_raw = raw.get("reward", {})
        sim_raw    = raw.get("simulation", {})
        city   = CityConfig(**{k: v for k, v in city_raw.items()   if hasattr(CityConfig(), k)})
        reward = RewardConfig(**{k: v for k, v in reward_raw.items() if hasattr(RewardConfig(), k)})
        # The shared-memory prefix must match the C++ server's --prefix. An env var
        # lets launch scripts give each run a unique prefix so concurrent servers
        # (e.g. a training run alongside the dashboard) don't collide on /dev/shm.
        shm_prefix = os.environ.get("TRAFFICRL_SHM_PREFIX") or raw.get("shm_prefix", "trafficrl")
        return cls(
            city=city,
            reward=reward,
            observation_mode=raw.get("observation_mode", "perfect"),
            communication_mode=raw.get("communication_mode", "none"),
            episode_length_steps=sim_raw.get("episode_length_steps", 2000),
            dt=sim_raw.get("dt", 0.1),
            shm_prefix=shm_prefix,
        )
