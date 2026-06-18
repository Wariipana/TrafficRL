from __future__ import annotations
import numpy as np
import functools
from typing import Any

import gymnasium
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

from .data_types import EnvConfig, GraphData, StateSnapshot, MAX_LANES
from .bridge_client import BridgeClient
from .spaces import (
    MAX_VEH_PER_LANE, MAX_SPEED_MS, MAX_WAIT_S, MAX_PHASE_TIME_S,
)
from .reward import compute_reward


class MARLTrafficEnv(ParallelEnv):
    """
    Multi-agent traffic environment (PettingZoo ParallelEnv).

    One agent per traffic light: agent_i controls intersection i.
    Agents act simultaneously; observations are local + 1-hop neighbor summary.

    Agent names: "light_0", "light_1", ..., "light_{n-1}"
    Observation: Dict with local intersection obs + neighbor_summary vector
    Action: Discrete(2) — 0=NS_GREEN, 1=EW_GREEN
    """

    metadata = {"name": "marl_traffic_v1", "render_modes": ["human"], "render_fps": 10}

    def __init__(self, config: EnvConfig, render_mode: str | None = None):
        super().__init__()
        self.config      = config
        self.render_mode = render_mode
        self._bridge     = BridgeClient(shm_prefix=config.shm_prefix)
        self._graph: GraphData | None    = None
        self._last_state: StateSnapshot | None = None
        self._episode_seed = config.city.seed
        self._adj: list[list[int]] = []   # adj[i] = list of neighbor light indices

        # These are set lazily after first connect
        self.possible_agents: list[str] = []
        self.agents:          list[str] = []

    # ---- PettingZoo API ----

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gymnasium.Space:
        if self._graph is None:
            raise RuntimeError("Call reset() before accessing observation_space.")
        return self._build_obs_space()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gymnasium.Space:
        return gymnasium.spaces.Discrete(2)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._episode_seed = seed

        if self._graph is None:
            graph = self._bridge.connect()
        else:
            graph = self._bridge.reset_episode(self._episode_seed)

        self._graph = graph
        self._adj   = self._build_adjacency(graph)

        n = graph.num_lights
        self.possible_agents = [f"light_{i}" for i in range(n)]
        self.agents          = list(self.possible_agents)

        # Invalidate cached spaces after topology may have changed
        self.observation_space.cache_clear()
        self.action_space.cache_clear()

        state = self._bridge.wait_for_state()
        self._last_state = state

        obs  = self._state_to_obs_all(state)
        info = {ag: self._intersection_info(state, i)
                for i, ag in enumerate(self.agents)}
        return obs, info

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict, dict, dict, dict, dict]:
        n      = self._graph.num_lights
        phases = np.zeros(n, dtype=np.uint8)
        for i, ag in enumerate(self.agents):
            if ag in actions:
                phases[i] = int(actions[ag]) & 0x1

        self._bridge.send_action(phases)
        state = self._bridge.wait_for_state()
        self._last_state = state

        obs        = self._state_to_obs_all(state)
        shared_r   = compute_reward(state, self.config.reward)

        rewards     = {}
        terminations = {}
        truncations  = {}
        infos        = {}

        for i, ag in enumerate(self.agents):
            rewards[ag]      = self._local_reward(state, i, shared_r)
            terminations[ag] = state.terminated
            truncations[ag]  = state.truncated
            infos[ag]        = self._intersection_info(state, i)

        if state.terminated or state.truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def close(self) -> None:
        self._bridge.disconnect()
        self._graph = None

    # ---- Observation helpers ----

    def _build_obs_space(self) -> gymnasium.spaces.Dict:
        n_neighbors = max(len(nb) for nb in self._adj) if self._adj else 4
        return gymnasium.spaces.Dict({
            "vehicles_per_lane": gymnasium.spaces.Box(
                0.0, MAX_VEH_PER_LANE, shape=(MAX_LANES,), dtype=np.float32),
            "queue_length": gymnasium.spaces.Box(
                0.0, MAX_VEH_PER_LANE, shape=(MAX_LANES,), dtype=np.float32),
            "avg_speed": gymnasium.spaces.Box(
                0.0, MAX_SPEED_MS, shape=(MAX_LANES,), dtype=np.float32),
            "avg_wait_time": gymnasium.spaces.Box(
                0.0, MAX_WAIT_S, shape=(1,), dtype=np.float32),
            "current_phase": gymnasium.spaces.Discrete(2),
            "phase_timer": gymnasium.spaces.Box(
                0.0, MAX_PHASE_TIME_S, shape=(1,), dtype=np.float32),
            # Mean neighbor pressure: [avg_queue, avg_wait, avg_throughput]
            "neighbor_summary": gymnasium.spaces.Box(
                0.0, 1.0, shape=(3,), dtype=np.float32),
        })

    def _state_to_obs_all(self, state: StateSnapshot) -> dict[str, dict]:
        n = self._graph.num_lights
        obs = {}
        for i, ag in enumerate(self.agents):
            obs[ag] = self._obs_for_intersection(state, i)
        return obs

    def _obs_for_intersection(self, state: StateSnapshot, idx: int) -> dict:
        s = state.intersections[idx]

        # Neighbor summary: average normalized queue, wait, throughput
        neighbors = self._adj[idx]
        if neighbors:
            nbs = [state.intersections[j] for j in neighbors]
            nb_queue  = float(np.mean([np.mean(nb.queue_length) for nb in nbs])) / MAX_VEH_PER_LANE
            nb_wait   = float(np.mean([nb.avg_wait_time for nb in nbs])) / MAX_WAIT_S
            nb_tp     = float(np.mean([nb.throughput for nb in nbs])) / 50.0
        else:
            nb_queue = nb_wait = nb_tp = 0.0

        return {
            "vehicles_per_lane": s.vehicles_per_lane[:MAX_LANES].astype(np.float32),
            "queue_length":      s.queue_length[:MAX_LANES].astype(np.float32),
            "avg_speed":         s.avg_speed[:MAX_LANES].astype(np.float32),
            "avg_wait_time":     np.array([s.avg_wait_time], dtype=np.float32),
            "current_phase":     int(s.phase),
            "phase_timer":       np.array([s.phase_timer_s], dtype=np.float32),
            "neighbor_summary":  np.array(
                [np.clip(nb_queue, 0, 1), np.clip(nb_wait, 0, 1), np.clip(nb_tp, 0, 1)],
                dtype=np.float32,
            ),
        }

    # ---- Reward ----

    def _local_reward(self, state: StateSnapshot, idx: int, global_r: float) -> float:
        s         = state.intersections[idx]
        wait_norm = float(np.clip(s.avg_wait_time / 600.0, 0.0, 1.0))
        tp_norm   = float(np.clip(s.throughput    /  50.0, 0.0, 1.0))
        return -wait_norm + 0.1 * tp_norm

    # ---- Info ----

    def _intersection_info(self, state: StateSnapshot, idx: int) -> dict:
        s = state.intersections[idx]
        return {
            "sim_tick":    state.sim_tick,
            "phase":       s.phase,
            "queue_total": float(np.sum(s.queue_length)),
            "throughput":  s.throughput,
            "avg_wait":    s.avg_wait_time,
        }

    # ---- Graph helpers ----

    def _build_adjacency(self, graph: GraphData) -> list[list[int]]:
        """
        Build per-light-index neighbor lists from graph edges.
        Only light→light adjacency is tracked.
        """
        n           = graph.num_lights
        light_ids   = set(graph.light_node_ids)
        node_to_idx = {nid: i for i, nid in enumerate(graph.light_node_ids)}

        adj: list[list[int]] = [[] for _ in range(n)]
        for edge in graph.edges:
            src, dst = edge.from_node, edge.to_node
            if src in node_to_idx and dst in node_to_idx:
                i, j = node_to_idx[src], node_to_idx[dst]
                if j not in adj[i]:
                    adj[i].append(j)
                if i not in adj[j]:
                    adj[j].append(i)

        return adj
