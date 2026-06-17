"""
Training entry point for the multi-agent IPPO + GNN agent.

Usage:
    python -m rl.training.train_ippo --config config/city_small.yaml
    python -m rl.training.train_ippo --config config/city_medium.yaml --steps 1000000

Requires a running trafficrl_server (it creates the shared memory the env reads).
The trained model is saved to <save-path>.pt automatically.
"""
from __future__ import annotations
import argparse

from rl.env.data_types import EnvConfig
from rl.agents.marl.ippo_agent import IPPOConfig, train_ippo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IPPO + GNN agent on TrafficRL")
    parser.add_argument("--config",    default="config/city_small.yaml", help="Env config YAML")
    parser.add_argument("--steps",     type=int, default=None,            help="Total timesteps (overrides default)")
    parser.add_argument("--log-dir",   default="rl/runs/ippo",            help="TensorBoard log dir")
    parser.add_argument("--save-path", default="rl/models/ippo_gnn",      help="Model save path (.pt appended)")
    parser.add_argument("--seed",      type=int, default=42,              help="Random seed")
    parser.add_argument("--device",    default="cpu",                     help="cpu | cuda")
    args = parser.parse_args()

    env_cfg = EnvConfig.from_yaml(args.config)

    cfg = IPPOConfig(
        env             = env_cfg,
        total_timesteps = args.steps or IPPOConfig.total_timesteps,
        log_dir         = args.log_dir,
        save_path       = args.save_path,
        seed            = args.seed,
        device          = args.device,
    )

    print(f"[train_ippo] Config: {args.config}")
    print(f"[train_ippo] Steps:  {cfg.total_timesteps}")
    train_ippo(cfg)


if __name__ == "__main__":
    main()
