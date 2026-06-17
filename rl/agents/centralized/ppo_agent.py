from __future__ import annotations
import os
from dataclasses import dataclass, field

import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
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
    # Tuned across real runs (4x4 then 8x8):
    #  pass 1 (3e-4, kl 0.03):  approx_kl ~0.12, clip ~0.6 — unstable.
    #  pass 2 (1e-4, kl 0.03):  KL healthy but reward FLAT — updates cut too early.
    #  pass 3 (2e-4, kl 0.06):  fine for 4x4; in 8x8 the per-update KL hits ~0.08
    #          with 56 agents, so target_kl=0.06 aborted the update at epoch ~2 and
    #          the policy stayed near-uniform (entropy ≈ max) → reward FLAT again.
    #  pass 4 (this): target_kl 0.06→0.12 and ent_coef 0.003→0.001. The KL grows
    #          fast with many agents, so a stricter target throttles the update;
    #          a looser cap + less entropy pressure lets the policy actually commit.
    #          With many agents the action-space entropy is large (56·ln2 ≈ 39), so
    #          even 0.001 keeps meaningful exploration.
    learning_rate:   float = 2e-4
    ent_coef:        float = 0.001
    target_kl:       float = 0.12      # stop a PPO update early if it diverges too far
    log_dir:         str  = "rl/runs"
    save_path:       str  = "rl/models/ppo_centralized"
    seed:            int  = 42


def make_env(config: EnvConfig):
    def _init():
        env = TrafficEnv(config)
        env = gymnasium.wrappers.FlattenObservation(env)
        # Monitor records per-episode reward/length so SB3 logs rollout/ep_rew_mean
        # — the key signal for "is the agent actually learning". Without it SB3
        # only prints train/ and time/, and the reward curve is invisible.
        env = Monitor(env)
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
        target_kl=config.target_kl,
        verbose=1,
        tensorboard_log=config.log_dir,
        seed=config.seed,
        policy_kwargs={"net_arch": [256, 256, 128]},
        # Force CPU: with a small MlpPolicy the GPU is slower (data-transfer
        # overhead dominates the tiny forward pass). SB3 warns about this too.
        device="cpu",
    )

    model.learn(total_timesteps=config.total_timesteps)
    model.save(config.save_path)
    env.save(config.save_path + "_vecnorm.pkl")
    env.close()

    print(f"[PPO] Model saved to {config.save_path}")
    return model


def load_model(save_path: str, env: gymnasium.Env | None = None) -> PPO:
    # CPU for inference too: the MlpPolicy is small and GPU transfer overhead
    # would only slow the per-step prediction during visualization.
    return PPO.load(save_path, env=env, device="cpu")
