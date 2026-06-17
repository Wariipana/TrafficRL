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

def _snapshot_to_episode_metrics(
    state: StateSnapshot,
    ep_id:        int,
    steps:        int,
    ep_max_wait:  float,
    ep_throughput_acc: float,
) -> EpisodeMetrics:
    avg_speed = float(np.mean([
        np.mean(s.avg_speed[:s.num_lanes]) if s.num_lanes > 0
        else 0.0
        for s in state.intersections
    ])) if state.intersections else 0.0

    return EpisodeMetrics(
        total_vehicles_served = ep_throughput_acc,
        throughput_per_step   = ep_throughput_acc / max(steps, 1),
        avg_wait_time_s       = state.avg_wait_global,
        max_wait_time_s       = ep_max_wait,
        congestion_spread     = state.congestion_spread,
        avg_speed_ms          = avg_speed,
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


# ---- Fixed-time baseline (no server needed if env is injected) ----

class FixedTimeRunner(BaseRunner):
    """
    Alternates NS_GREEN / EW_GREEN every `cycle_steps` steps.
    Requires a running C++ server.
    """
    name = "fixed_time"

    def __init__(self, env_cfg: EnvConfig, cycle_steps: int = 30):
        import gymnasium
        from rl.env.traffic_env import TrafficEnv
        self._env_cfg     = env_cfg
        self._cycle_steps = cycle_steps
        self._env = gymnasium.wrappers.FlattenObservation(TrafficEnv(env_cfg))

    def _run_one_episode(self, seed: int) -> EpisodeMetrics:
        obs, _ = self._env.reset(seed=seed)
        done   = False
        step   = 0
        ep_max_wait = 0.0
        ep_tp_acc   = 0.0
        last_info: dict = {}

        n_lights = self._env.action_space.nvec.shape[0]

        while not done:
            phase  = (step // self._cycle_steps) % 2
            action = np.full(n_lights, phase, dtype=np.int64)
            obs, _, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            step += 1
            ep_max_wait  = max(ep_max_wait, info.get("max_wait_global", 0.0))
            ep_tp_acc   += info.get("total_throughput", 0.0)
            last_info    = info

        state = self._env.unwrapped._last_state
        return _snapshot_to_episode_metrics(state, seed, step, ep_max_wait, ep_tp_acc)

    def close(self) -> None:
        self._env.close()


# ---- Random policy baseline ----

class RandomRunner(BaseRunner):
    """Uniformly random actions — sets a lower bound on performance."""
    name = "random"

    def __init__(self, env_cfg: EnvConfig):
        import gymnasium
        from rl.env.traffic_env import TrafficEnv
        self._env = gymnasium.wrappers.FlattenObservation(TrafficEnv(env_cfg))

    def _run_one_episode(self, seed: int) -> EpisodeMetrics:
        rng    = np.random.default_rng(seed)
        obs, _ = self._env.reset(seed=seed)
        done   = False
        step   = 0
        ep_max_wait = 0.0
        ep_tp_acc   = 0.0

        while not done:
            action = self._env.action_space.sample()
            obs, _, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            step += 1
            ep_max_wait  = max(ep_max_wait, info.get("max_wait_global", 0.0))
            ep_tp_acc   += info.get("total_throughput", 0.0)

        state = self._env.unwrapped._last_state
        return _snapshot_to_episode_metrics(state, seed, step, ep_max_wait, ep_tp_acc)

    def close(self) -> None:
        self._env.close()


# ---- Centralized PPO ----

class PPOCentralizedRunner(BaseRunner):
    """Loads a saved SB3 PPO model and evaluates it."""
    name = "ppo_centralized"

    def __init__(self, env_cfg: EnvConfig, model_path: str):
        import gymnasium
        from rl.env.traffic_env import TrafficEnv
        from rl.agents.centralized.ppo_agent import load_model
        self._env   = gymnasium.wrappers.FlattenObservation(TrafficEnv(env_cfg))
        self._model = load_model(model_path, self._env)

    def _run_one_episode(self, seed: int) -> EpisodeMetrics:
        obs, _ = self._env.reset(seed=seed)
        done   = False
        step   = 0
        ep_max_wait = 0.0
        ep_tp_acc   = 0.0

        while not done:
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            step += 1
            ep_max_wait  = max(ep_max_wait, info.get("max_wait_global", 0.0))
            ep_tp_acc   += info.get("total_throughput", 0.0)

        state = self._env.unwrapped._last_state
        return _snapshot_to_episode_metrics(state, seed, step, ep_max_wait, ep_tp_acc)

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

        state = self._env._last_state
        return _snapshot_to_episode_metrics(state, seed, step, ep_max_wait, ep_tp_acc)

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

        state = self._env._last_state
        return _snapshot_to_episode_metrics(state, seed, step, ep_max_wait, ep_tp_acc)

    def close(self) -> None:
        self._env.close()
