"""
Training entry point.

Usage:
    python -m rl.training.train --config config/city_small.yaml
    python -m rl.training.train --config config/city_medium.yaml --steps 1000000
"""
from __future__ import annotations
import argparse

from rl.env.data_types import EnvConfig
from rl.agents.centralized.ppo_agent import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train centralized PPO agent on TrafficRL")
    parser.add_argument("--config",    default="config/city_small.yaml", help="Env config YAML")
    parser.add_argument("--steps",     type=int,   default=None,         help="Total timesteps (overrides config)")
    parser.add_argument("--log-dir",   default="rl/runs",                help="TensorBoard log dir")
    parser.add_argument("--save-path", default="rl/models/ppo_centralized", help="Model save path")
    parser.add_argument("--seed",      type=int, default=42,             help="Random seed")
    args = parser.parse_args()

    env_cfg = EnvConfig.from_yaml(args.config)

    train_cfg = TrainingConfig(
        env           = env_cfg,
        total_timesteps = args.steps or 1_000_000,
        log_dir       = args.log_dir,
        save_path     = args.save_path,
        seed          = args.seed,
    )

    print(f"[train] Config: {args.config}")
    print(f"[train] Steps:  {train_cfg.total_timesteps}")
    train(train_cfg)


if __name__ == "__main__":
    main()
