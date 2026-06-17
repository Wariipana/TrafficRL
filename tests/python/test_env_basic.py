"""
Basic TrafficEnv tests that run without the C++ server.
These test Python-side logic only (data_types, spaces, reward).
Full integration tests (requiring the C++ server) are in test_shared_memory.py.
"""
from __future__ import annotations
import numpy as np
import pytest

from rl.env.data_types import (
    EnvConfig, CityConfig, RewardConfig,
    GraphData, NodeRecord, EdgeRecord,
    StateSnapshot, IntersectionSnapshot,
    MAX_LANES,
)
from rl.env.spaces import build_observation_space, build_action_space
from rl.env.reward import compute_reward


def _make_graph(n_lights: int = 4) -> GraphData:
    nodes = [
        NodeRecord(id=i, x=float(i*100), y=0.0, zone=0,
                   has_light=True, light_id=i, num_outgoing=2)
        for i in range(n_lights)
    ]
    edges = [
        EdgeRecord(id=i, from_node=i, to_node=(i+1)%n_lights,
                   length=100.0, num_lanes=2, speed_limit=50/3.6, direction=2)
        for i in range(n_lights)
    ]
    return GraphData(
        nodes=nodes, edges=edges,
        light_node_ids=list(range(n_lights)),
        num_lights=n_lights,
    )


def _make_state(n_lights: int = 4) -> StateSnapshot:
    intersections = [
        IntersectionSnapshot(
            id=i, phase=0, num_lanes=2, phase_timer_s=5.0,
            vehicles_per_lane=np.zeros(MAX_LANES, dtype=np.float32),
            queue_length=np.zeros(MAX_LANES, dtype=np.float32),
            avg_speed=np.zeros(MAX_LANES, dtype=np.float32),
            avg_wait_time=10.0,
            throughput=2.0,
        )
        for i in range(n_lights)
    ]
    return StateSnapshot(
        sim_tick=1, num_intersections=n_lights, num_vehicles=10,
        sim_time_s=1.0, episode_step=1,
        terminated=False, truncated=False,
        intersections=intersections,
        total_throughput=8.0,
        avg_wait_global=10.0,
        max_wait_global=20.0,
        congestion_spread=0.2,
    )


def test_observation_space_shapes():
    graph = _make_graph(n_lights=4)
    cfg   = EnvConfig()
    space = build_observation_space(graph, cfg)
    assert space["vehicles_per_lane"].shape == (4, MAX_LANES)
    assert space["queue_length"].shape      == (4, MAX_LANES)
    assert space["avg_speed"].shape         == (4, MAX_LANES)
    assert space["avg_wait_time"].shape     == (4,)
    assert space["current_phase"].nvec.shape == (4,)
    assert space["phase_timer"].shape       == (4,)


def test_action_space_shape():
    graph  = _make_graph(n_lights=6)
    cfg    = EnvConfig()
    aspace = build_action_space(graph)
    assert aspace.nvec.shape == (6,)
    assert all(n == 2 for n in aspace.nvec)


def test_observation_space_contains_valid_obs():
    graph = _make_graph(n_lights=4)
    cfg   = EnvConfig()
    space = build_observation_space(graph, cfg)
    sample = space.sample()
    assert space.contains(sample)


def test_reward_negative_without_control():
    graph = _make_graph(n_lights=4)
    cfg   = EnvConfig()
    state = _make_state(n_lights=4)
    state.total_throughput = 0.0  # no throughput
    reward = compute_reward(state, cfg.reward)
    assert reward < 0.0, f"Expected negative reward without throughput, got {reward}"


def test_reward_improves_with_throughput():
    graph = _make_graph(n_lights=4)
    cfg   = EnvConfig()

    state_bad  = _make_state(n_lights=4)
    state_good = _make_state(n_lights=4)
    state_good.total_throughput = 100.0
    for s in state_good.intersections:
        s.avg_wait_time = 0.0
        s.queue_length  = np.zeros(MAX_LANES, dtype=np.float32)
        s.throughput    = 10.0

    r_bad  = compute_reward(state_bad,  cfg.reward)
    r_good = compute_reward(state_good, cfg.reward)
    assert r_good > r_bad, "Better state should yield higher reward"


def test_env_config_from_yaml(tmp_path):
    yaml_content = """
city:
  grid_width: 4
  grid_height: 4
  seed: 99
simulation:
  episode_length_steps: 500
  dt: 0.1
observation_mode: perfect
communication_mode: none
shm_prefix: trafficrl
"""
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content)
    cfg = EnvConfig.from_yaml(str(p))
    assert cfg.city.grid_width == 4
    assert cfg.city.seed == 99
    assert cfg.episode_length_steps == 500


def test_reward_config_defaults():
    cfg = RewardConfig()
    assert 0.0 < cfg.alpha < 1.0
    assert abs(cfg.local_weight + cfg.global_weight - 1.0) < 0.01
