from __future__ import annotations
import os
from dataclasses import dataclass, field

import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.env.traffic_env import TrafficEnv
from rl.env.data_types import EnvConfig


@dataclass
class TrainingConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    total_timesteps: int = 1_000_000
    n_steps:         int = 2048
    batch_size:      int = 64
    n_epochs:        int = 10
    learning_rate:   float = 3e-4
    ent_coef:        float = 0.01
    log_dir:         str  = "rl/runs"
    save_path:       str  = "rl/models/ppo_centralized"
    seed:            int  = 42


def make_env(config: EnvConfig):
    def _init():
        env = TrafficEnv(config)
        env = gymnasium.wrappers.FlattenObservation(env)
        return env
    return _init


def train(config: TrainingConfig) -> PPO:
    os.makedirs(config.log_dir,  exist_ok=True)
    os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)

    env = DummyVecEnv([make_env(config.env)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        ent_coef=config.ent_coef,
        verbose=1,
        tensorboard_log=config.log_dir,
        seed=config.seed,
        policy_kwargs={"net_arch": [256, 256, 128]},
    )

    model.learn(total_timesteps=config.total_timesteps)
    model.save(config.save_path)
    env.save(config.save_path + "_vecnorm.pkl")
    env.close()

    print(f"[PPO] Model saved to {config.save_path}")
    return model


def load_model(save_path: str, env: gymnasium.Env | None = None) -> PPO:
    return PPO.load(save_path, env=env)
