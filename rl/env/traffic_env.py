from __future__ import annotations
import numpy as np
import gymnasium
from gymnasium.utils import passive_env_checker

from .data_types import EnvConfig, GraphData, StateSnapshot, MAX_LANES
from .bridge_client import BridgeClient
from .spaces import build_observation_space, build_action_space
from .reward import compute_reward


class TrafficEnv(gymnasium.Env):
    """
    Gymnasium environment for traffic signal control.
    Connects to a running trafficrl_server via POSIX shared memory.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, config: EnvConfig, render_mode: str | None = None):
        super().__init__()
        self.config      = config
        self.render_mode = render_mode
        self._bridge     = BridgeClient(shm_prefix=config.shm_prefix)
        self._graph: GraphData | None = None
        self._last_state: StateSnapshot | None = None
        self._episode_seed = config.city.seed

        # Spaces depend on the graph topology, so they require a live connection
        # to the server. Connect eagerly here so observation_space/action_space
        # are populated right after construction — wrappers such as
        # FlattenObservation inspect them before the first reset() and break if
        # they are still None.
        self.observation_space: gymnasium.Space | None = None
        self.action_space:      gymnasium.Space | None = None
        self._ensure_connected()

    def _ensure_connected(self) -> None:
        if self._graph is not None:
            return
        graph = self._bridge.connect()
        self._graph            = graph
        self.observation_space = build_observation_space(graph, self.config)
        self.action_space      = build_action_space(graph)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ensure_connected()

        if seed is not None:
            self._episode_seed = seed

        graph = self._bridge.reset_episode(self._episode_seed)
        # Update spaces if graph topology changed (rare, but possible with different seeds)
        self._graph            = graph
        self.observation_space = build_observation_space(graph, self.config)
        self.action_space      = build_action_space(graph)

        state = self._bridge.wait_for_state()
        self._last_state = state

        obs  = self._state_to_obs(state)
        info = self._state_to_info(state)
        return obs, info

    def step(self, action):
        phases = np.asarray(action, dtype=np.uint8)
        self._bridge.send_action(phases)
        state = self._bridge.wait_for_state()
        self._last_state = state

        obs        = self._state_to_obs(state)
        reward     = compute_reward(state, self.config.reward)
        terminated = state.terminated
        truncated  = state.truncated
        info       = self._state_to_info(state)

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._bridge.disconnect()
        self._graph = None

    # ---- Observation / info helpers ----

    def _state_to_obs(self, state: StateSnapshot) -> dict:
        n = self._graph.num_lights

        vpl  = np.zeros((n, MAX_LANES), dtype=np.float32)
        ql   = np.zeros((n, MAX_LANES), dtype=np.float32)
        spd  = np.zeros((n, MAX_LANES), dtype=np.float32)
        wait = np.zeros(n, dtype=np.float32)
        phase = np.zeros(n, dtype=np.int64)
        timer = np.zeros(n, dtype=np.float32)

        for i, s in enumerate(state.intersections[:n]):
            vpl[i]   = s.vehicles_per_lane[:MAX_LANES]
            ql[i]    = s.queue_length[:MAX_LANES]
            spd[i]   = s.avg_speed[:MAX_LANES]
            wait[i]  = s.avg_wait_time
            phase[i] = s.phase
            timer[i] = s.phase_timer_s

        return {
            "vehicles_per_lane": vpl,
            "queue_length":      ql,
            "avg_speed":         spd,
            "avg_wait_time":     wait,
            "current_phase":     phase,
            "phase_timer":       timer,
        }

    def _state_to_info(self, state: StateSnapshot) -> dict:
        return {
            "sim_tick":          state.sim_tick,
            "num_vehicles":      state.num_vehicles,
            "avg_wait_global":   state.avg_wait_global,
            "max_wait_global":   state.max_wait_global,
            "total_throughput":  state.total_throughput,
            "congestion_spread": state.congestion_spread,
            "episode_step":      state.episode_step,
        }
