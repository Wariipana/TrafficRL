"""
Benchmark runners — one per algorithm type.

All runners implement run_episodes(n_episodes, seed_offset) → AlgorithmResult
and share a common interface so benchmark.py can call them uniformly.

Runners that require a live C++ server will raise RuntimeError if the server
is not running; runners for mock / random policies work offline.
"""
from __future__ import annotations
import time
from typing import Optional
import numpy as np

from rl.env.data_types import EnvConfig, StateSnapshot, MAX_LANES
from rl.benchmark.metrics import EpisodeMetrics, AlgorithmResult


# ---- helpers ----

def episode_speed(state: StateSnapshot) -> float:
    """Mean per-lane speed across all active lanes in a single snapshot."""
    return float(np.mean([
        np.mean(s.avg_speed[:s.num_lanes]) if s.num_lanes > 0
        else 0.0
        for s in state.intersections
    ])) if state.intersections else 0.0


def _episode_metrics(
    ep_id:             int,
    steps:             int,
    ep_max_wait:       float,
    ep_throughput_acc: float,
    ep_wait_acc:       float,
    ep_congestion_acc: float,
    ep_speed_acc:      float,
) -> EpisodeMetrics:
    # Wait/congestion/speed are averaged over the WHOLE episode, not read from the
    # last snapshot. A single final-step reading is noisy and unfair: it rewards a
    # policy that happens to leave the grid empty on the last step and punishes one
    # that doesn't, regardless of how the episode actually went.
    inv = 1.0 / max(steps, 1)
    return EpisodeMetrics(
        total_vehicles_served = ep_throughput_acc,
        throughput_per_step   = ep_throughput_acc * inv,
        avg_wait_time_s       = ep_wait_acc       * inv,
        max_wait_time_s       = ep_max_wait,
        congestion_spread     = ep_congestion_acc * inv,
        avg_speed_ms          = ep_speed_acc      * inv,
        steps_completed       = steps,
        episode_id            = ep_id,
    )


# ---- Base class ----

class BaseRunner:
    name: str = "base"

    def run_episodes(
        self,
        n_episodes: int,
        seed_offset: int = 0,
        config_label: str = "unknown",
    ) -> AlgorithmResult:
        result = AlgorithmResult(name=self.name, config_label=config_label)
        for ep in range(n_episodes):
            m = self._run_one_episode(ep + seed_offset)
            result.episodes.append(m)
        result.compute_summary()
        return result

    def _run_one_episode(self, seed: int) -> EpisodeMetrics:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ---- Misconfigured fixed-time baseline ----

class FixedRandomRunner(BaseRunner):
    """
    "Badly misconfigured city" baseline: very long asymmetric fixed-time cycles
    that represent a common real-world failure mode — lights are set with
    a heavy bias toward one phase, leaving cross-traffic waiting for most
    of each cycle.

    Each light gets:
      - A very long random period (800-1200 steps per full cycle)
      - A minimal green window for phase 1 (10-15% of the period)
      - A random offset so the lights are unsynchronised with each other

    Effect: phase-1 vehicles (N-S direction) wait up to ~100 simulated seconds
    for their short green window while phase-0 (E-W) monopolises the intersection
    for ~85-90% of each cycle.  This produces the high avg_wait and low throughput
    that RL must beat.
    """
    name = "fixed_random"

    def __init__(self, env_cfg: EnvConfig, period_min: int = 800, period_max: int = 1200):
        import gymnasium
        from rl.env.traffic_env import TrafficEnv
        self._env_cfg    = env_cfg
        self._period_min = period_min
        self._period_max = period_max
        self._env = gymnasium.wrappers.FlattenObservation(TrafficEnv(env_cfg))

    def _run_one_episode(self, seed: int) -> EpisodeMetrics:
        rng    = np.random.default_rng(seed)
        obs, _ = self._env.reset(seed=seed)
        done   = False
        step   = 0
        ep_max_wait = 0.0
        ep_tp_acc   = 0.0
        ep_wait_acc = 0.0
        ep_cong_acc = 0.0
        ep_spd_acc  = 0.0

        n_lights = self._env.action_space.nvec.shape[0]
        # Heavily asymmetric long cycles compatible with the motor's min_green
        # constraint (DEFAULT_MIN_GREEN = 10 s = 100 steps at dt=0.1 s).
        #   period 800-1200 steps → full cycle = 80-120 simulated seconds
        #   phase 1 (N-S) = 10-15% → 100-180 steps ≥ min_green ✓
        #   phase 0 (E-W) = 85-90% → 700-1080 steps — N-S vehicles wait up to
        #   ~100 simulated seconds per cycle, producing high avg_wait metrics.
        # Random per-light periods and offsets keep the lights unsynchronised so
        # there is no accidental green-wave helping throughput.
        periods   = rng.integers(self._period_min, self._period_max + 1, size=n_lights)
        green_dur = np.maximum(
            100,  # hard floor = min_green; motor rejects switches before this
            (periods * rng.uniform(0.10, 0.15, size=n_lights)).astype(np.int64),
        )
        offsets = np.array([rng.integers(0, p) for p in periods], dtype=np.int64)

        while not done:
            pos_in_cycle = (step + offsets) % periods
            # Phase 1 active only during the last green_dur steps of each cycle.
            phase  = (pos_in_cycle >= (periods - green_dur)).astype(np.int64)
            action = phase
            obs, _, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            step += 1
            ep_max_wait  = max(ep_max_wait, info.get("max_wait_global", 0.0))
            ep_tp_acc   += info.get("total_throughput", 0.0)
            ep_wait_acc += info.get("avg_wait_global", 0.0)
            ep_cong_acc += info.get("congestion_spread", 0.0)
            ep_spd_acc  += episode_speed(self._env.unwrapped._last_state)

        return _episode_metrics(seed, step, ep_max_wait, ep_tp_acc,
                                ep_wait_acc, ep_cong_acc, ep_spd_acc)

    def close(self) -> None:
        self._env.close()


# ---- Centralized PPO ----

class PPOCentralizedRunner(BaseRunner):
    """Loads a saved SB3 PPO model and evaluates it."""
    name = "ppo_centralized"

    def __init__(self, env_cfg: EnvConfig, model_path: str):
        import os
        import pickle
        import gymnasium
        from rl.env.traffic_env import TrafficEnv
        from rl.agents.centralized.ppo_agent import load_model

        self._env = gymnasium.wrappers.FlattenObservation(TrafficEnv(env_cfg))

        # Load the VecNormalize statistics saved alongside the model.
        # Without these the model receives raw observations at evaluation time,
        # which do not match the normalised distribution it was trained on.
        # We load the pickle directly (SB3 excludes the venv from the pickle via
        # __getstate__) so no second TrafficEnv is created.
        self._vecnorm = None
        vecnorm_path = model_path + "_vecnorm.pkl"
        if os.path.exists(vecnorm_path):
            with open(vecnorm_path, "rb") as f:
                self._vecnorm = pickle.load(f)

        # Load without passing env: avoids SB3 re-wrapping self._env in a
        # Monitor + DummyVecEnv which would open a competing shm connection.
        self._model = load_model(model_path)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply VecNormalize obs statistics if available."""
        if self._vecnorm is None:
            return obs
        vn  = self._vecnorm
        obs = (obs - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon)
        return np.clip(obs, -vn.clip_obs, vn.clip_obs).astype(np.float32)

    def _run_one_episode(self, seed: int) -> EpisodeMetrics:
        obs, _ = self._env.reset(seed=seed)
        done   = False
        step   = 0
        ep_max_wait = 0.0
        ep_tp_acc   = 0.0
        ep_wait_acc = 0.0
        ep_cong_acc = 0.0
        ep_spd_acc  = 0.0

        while not done:
            action, _ = self._model.predict(self._normalize_obs(obs), deterministic=True)
            obs, _, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            step += 1
            ep_max_wait  = max(ep_max_wait, info.get("max_wait_global", 0.0))
            ep_tp_acc   += info.get("total_throughput", 0.0)
            ep_wait_acc += info.get("avg_wait_global", 0.0)
            ep_cong_acc += info.get("congestion_spread", 0.0)
            ep_spd_acc  += episode_speed(self._env.unwrapped._last_state)

        return _episode_metrics(seed, step, ep_max_wait, ep_tp_acc,
                                ep_wait_acc, ep_cong_acc, ep_spd_acc)

    def close(self) -> None:
        self._env.close()


# ---- IPPO + GNN ----

class IPPOGNNRunner(BaseRunner):
    """Evaluates a saved IPPOActorCritic (Fase 4) model."""
    name = "ippo_gnn"

    def __init__(self, env_cfg: EnvConfig, model_path: str, k_hops: int = 1):
        import torch
        from rl.env.marl_env import MARLTrafficEnv
        from rl.agents.marl.ippo_agent import IPPOActorCritic, flatten_obs, LOCAL_FEAT_DIM
        from rl.models.gnn import build_adjacency_mask

        self._env   = MARLTrafficEnv(env_cfg)
        obs_d, _    = self._env.reset()
        n           = self._env._graph.num_lights

        self._model = IPPOActorCritic()
        self._model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self._model.eval()

        self._adj = build_adjacency_mask(
            self._env._graph.light_node_ids,
            self._env._graph.edges,
            k_hops=k_hops,
        )
        self._n         = n
        self._flatten   = flatten_obs
        self._torch     = torch
        # Reset again with proper seed later
        self._obs_d     = obs_d

    def _run_one_episode(self, seed: int) -> EpisodeMetrics:
        import torch
        from rl.agents.marl.ippo_agent import flatten_obs

        obs_d, _ = self._env.reset(seed=seed)
        done     = False
        step     = 0
        ep_max_wait = 0.0
        ep_tp_acc   = 0.0
        ep_wait_acc = 0.0
        ep_cong_acc = 0.0
        ep_spd_acc  = 0.0

        while not done:
            flat = torch.tensor(
                np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(self._n)]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                actions, _, _ = self._model.get_action(flat, self._adj, deterministic=True)
            action_dict = {f"light_{i}": int(actions[i]) for i in range(self._n)}

            obs_d, rews, terms, truncs, info_d = self._env.step(action_dict)
            done = not bool(self._env.agents)
            step += 1

            state = self._env._last_state
            ep_max_wait  = max(ep_max_wait, state.max_wait_global)
            ep_tp_acc   += state.total_throughput
            ep_wait_acc += state.avg_wait_global
            ep_cong_acc += state.congestion_spread
            ep_spd_acc  += episode_speed(state)

        return _episode_metrics(seed, step, ep_max_wait, ep_tp_acc,
                                ep_wait_acc, ep_cong_acc, ep_spd_acc)

    def close(self) -> None:
        self._env.close()


# ---- HRL (Manager + Worker) ----

class HRLRunner(BaseRunner):
    """Evaluates saved HRL Manager + Worker models."""
    name = "hrl"

    def __init__(
        self,
        env_cfg: EnvConfig,
        worker_path: str,
        manager_path: str,
        k_hops: int = 1,
    ):
        import torch
        from rl.env.marl_env import MARLTrafficEnv
        from rl.agents.hrl.worker import HRLWorkerActorCritic
        from rl.agents.hrl.manager import HRLManager, ManagerConfig
        from rl.models.gnn import build_adjacency_mask

        self._env    = MARLTrafficEnv(env_cfg)
        obs_d, _     = self._env.reset()
        n            = self._env._graph.num_lights
        num_zones    = min(8, max(1, n // 4))

        self._worker = HRLWorkerActorCritic()
        self._worker.load_state_dict(torch.load(worker_path, map_location="cpu"))
        self._worker.eval()

        self._manager = HRLManager(ManagerConfig(), num_zones=num_zones)
        self._manager.load(manager_path)

        self._adj = build_adjacency_mask(
            self._env._graph.light_node_ids,
            self._env._graph.edges,
            k_hops=k_hops,
        )
        self._n        = n
        self._nz       = num_zones
        self._torch    = torch

    def _run_one_episode(self, seed: int) -> EpisodeMetrics:
        import torch
        from rl.agents.marl.ippo_agent import flatten_obs

        obs_d, _ = self._env.reset(seed=seed)
        done     = False
        step     = 0
        ep_max_wait = 0.0
        ep_tp_acc   = 0.0
        ep_wait_acc = 0.0
        ep_cong_acc = 0.0
        ep_spd_acc  = 0.0

        while not done:
            state     = self._env._last_state
            goals_np  = self._manager.get_goals(state, step)
            goals_per = np.stack([
                goals_np[int(self._env._graph.light_node_ids[i]) % self._nz]
                for i in range(self._n)
            ])

            flat   = torch.tensor(
                np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(self._n)]),
                dtype=torch.float32,
            )
            goals_t = torch.tensor(goals_per, dtype=torch.float32)
            with torch.no_grad():
                actions, _, _ = self._worker.get_action(flat, goals_t, self._adj, deterministic=True)

            action_dict = {f"light_{i}": int(actions[i]) for i in range(self._n)}
            obs_d, _, terms, truncs, _ = self._env.step(action_dict)
            done = not bool(self._env.agents)
            step += 1

            state = self._env._last_state
            ep_max_wait  = max(ep_max_wait, state.max_wait_global)
            ep_tp_acc   += state.total_throughput
            ep_wait_acc += state.avg_wait_global
            ep_cong_acc += state.congestion_spread
            ep_spd_acc  += episode_speed(state)

        return _episode_metrics(seed, step, ep_max_wait, ep_tp_acc,
                                ep_wait_acc, ep_cong_acc, ep_spd_acc)

    def close(self) -> None:
        self._env.close()
