"""
Evaluation script — metrics are separate from the reward function (Goodhart's Law).

Usage:
    python -m rl.training.evaluate --model rl/models/ppo_centralized --config config/city_medium.yaml
    python -m rl.training.evaluate --baseline fixed_time --config config/city_medium.yaml
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass

import numpy as np
import gymnasium

from rl.env.traffic_env import TrafficEnv
from rl.env.data_types import EnvConfig
from rl.agents.centralized.ppo_agent import load_model


@dataclass
class EvalMetrics:
    avg_wait_time:      float
    max_wait_time:      float
    total_throughput:   float
    congestion_spread:  float
    n_episodes:         int

    def __str__(self) -> str:
        return (
            f"Episodes:          {self.n_episodes}\n"
            f"Avg wait time:     {self.avg_wait_time:.2f} s\n"
            f"Max wait time:     {self.max_wait_time:.2f} s\n"
            f"Total throughput:  {self.total_throughput:.1f} veh\n"
            f"Congestion spread: {self.congestion_spread:.3f}\n"
        )


def evaluate_agent(model, env: gymnasium.Env, n_episodes: int = 20) -> EvalMetrics:
    wait_times    = []
    max_waits     = []
    throughputs   = []
    congestions   = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_max_wait = 0.0
        ep_throughput = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_max_wait   = max(ep_max_wait, info.get("max_wait_global", 0.0))
            ep_throughput = info.get("total_throughput", 0.0)
            congestions.append(info.get("congestion_spread", 0.0))

        wait_times.append(info.get("avg_wait_global", 0.0))
        max_waits.append(ep_max_wait)
        throughputs.append(ep_throughput)

    return EvalMetrics(
        avg_wait_time    = float(np.mean(wait_times)),
        max_wait_time    = float(np.mean(max_waits)),
        total_throughput = float(np.mean(throughputs)),
        congestion_spread = float(np.mean(congestions)),
        n_episodes       = n_episodes,
    )


def evaluate_fixed_time_baseline(env: gymnasium.Env, n_episodes: int = 20,
                                  cycle_steps: int = 30) -> EvalMetrics:
    """Alternates NS/EW phases on a fixed cycle as baseline comparison."""
    wait_times  = []
    max_waits   = []
    throughputs = []
    congestions = []
    n_lights = env.action_space.nvec.shape[0] if hasattr(env.action_space, "nvec") else 1

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done    = False
        step    = 0
        ep_max_wait = 0.0

        while not done:
            phase  = (step // cycle_steps) % 2
            action = np.full(n_lights, phase, dtype=np.int64)
            obs, _, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            step += 1
            ep_max_wait = max(ep_max_wait, info.get("max_wait_global", 0.0))
            congestions.append(info.get("congestion_spread", 0.0))

        wait_times.append(info.get("avg_wait_global", 0.0))
        max_waits.append(ep_max_wait)
        throughputs.append(info.get("total_throughput", 0.0))

    return EvalMetrics(
        avg_wait_time    = float(np.mean(wait_times)),
        max_wait_time    = float(np.mean(max_waits)),
        total_throughput = float(np.mean(throughputs)),
        congestion_spread = float(np.mean(congestions)),
        n_episodes       = n_episodes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TrafficRL agent vs. baseline")
    parser.add_argument("--model",      default=None,                    help="Path to saved PPO model")
    parser.add_argument("--baseline",   default="fixed_time",            help="Baseline type: fixed_time")
    parser.add_argument("--config",     default="config/city_medium.yaml")
    parser.add_argument("--episodes",   type=int, default=20)
    args = parser.parse_args()

    env_cfg = EnvConfig.from_yaml(args.config)
    env     = gymnasium.wrappers.FlattenObservation(TrafficEnv(env_cfg))

    if args.model:
        print(f"\n=== Trained Agent: {args.model} ===")
        model = load_model(args.model, env)
        metrics = evaluate_agent(model, env, args.episodes)
        print(metrics)

    print(f"\n=== Baseline: {args.baseline} ===")
    baseline_metrics = evaluate_fixed_time_baseline(env, args.episodes)
    print(baseline_metrics)

    if args.model:
        ratio = metrics.avg_wait_time / (baseline_metrics.avg_wait_time + 1e-6)
        print(f"Wait time ratio (agent/baseline): {ratio:.3f} (target: <= 0.60)")

    env.close()


if __name__ == "__main__":
    main()
