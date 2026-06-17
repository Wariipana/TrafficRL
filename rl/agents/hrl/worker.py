from __future__ import annotations
"""
HRL Worker: extends IPPO with goal conditioning from the Manager.

The Worker receives, for each intersection, a goal vector [target_throughput,
target_wait_norm] from the Manager and appends it to the local observation.
This file provides:
  - GoalConditionedEncoder   — replaces LocalEncoder in IPPOActorCritic
  - HRLWorkerActorCritic     — IPPO model that accepts goals
  - goal_reward_shaping      — intrinsic reward for approaching Manager goals
"""
import numpy as np
import torch
import torch.nn as nn

from rl.agents.marl.ippo_agent import (
    IPPOActorCritic,
    LOCAL_FEAT_DIM,
    flatten_obs,
)
from rl.agents.hrl.manager import GOAL_DIM
from rl.models.gnn import TrafficGNN


WORKER_FEAT_DIM = LOCAL_FEAT_DIM + GOAL_DIM


class GoalConditionedEncoder(nn.Module):
    """Encodes [local_obs; goal_vector] into a dense feature vector."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WORKER_FEAT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HRLWorkerActorCritic(nn.Module):
    """
    Worker policy: identical to IPPOActorCritic but conditioned on Manager goals.

    Inputs:
        flat_obs_all:  (N, LOCAL_FEAT_DIM)  — local observations
        goals_all:     (N, GOAL_DIM)        — Manager goal vectors
        adj_mask:      (N, N) bool          — GNN adjacency

    Outputs:
        logits (N, 2), values (N,)
    """

    def __init__(
        self,
        local_enc_dim: int = 64,
        gnn_hidden:    int = 128,
        gnn_embed:     int = 64,
        actor_hidden:  int = 128,
        critic_hidden: int = 128,
    ):
        super().__init__()
        self.local_encoder = GoalConditionedEncoder(out_dim=local_enc_dim)
        self.gnn            = TrafficGNN(
            node_feat_dim=local_enc_dim,
            hidden_dim=gnn_hidden,
            embed_dim=gnn_embed,
        )
        fused_dim = local_enc_dim + gnn_embed

        self.actor = nn.Sequential(
            nn.Linear(fused_dim, actor_hidden),
            nn.Tanh(),
            nn.Linear(actor_hidden, actor_hidden),
            nn.Tanh(),
            nn.Linear(actor_hidden, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(fused_dim, critic_hidden),
            nn.Tanh(),
            nn.Linear(critic_hidden, critic_hidden),
            nn.Tanh(),
            nn.Linear(critic_hidden, 1),
        )

    def _encode(
        self,
        flat_obs_all: torch.Tensor,  # (N, LOCAL_FEAT_DIM)
        goals_all:    torch.Tensor,  # (N, GOAL_DIM)
        adj_mask:     torch.Tensor,  # (N, N) bool
    ) -> torch.Tensor:               # (N, fused_dim)
        combined   = torch.cat([flat_obs_all, goals_all], dim=-1)  # (N, WORKER_FEAT_DIM)
        local_feat = self.local_encoder(combined)                   # (N, enc_dim)
        gnn_feat   = self.gnn(local_feat, adj_mask)                 # (N, gnn_embed)
        return torch.cat([local_feat, gnn_feat], dim=-1)

    def forward(
        self,
        flat_obs_all: torch.Tensor,
        goals_all:    torch.Tensor,
        adj_mask:     torch.Tensor,
    ):
        fused  = self._encode(flat_obs_all, goals_all, adj_mask)
        logits = self.actor(fused)
        values = self.critic(fused).squeeze(-1)
        return logits, values

    def get_action(self, flat_obs_all, goals_all, adj_mask, deterministic=False):
        logits, values = self.forward(flat_obs_all, goals_all, adj_mask)
        dist   = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), values

    def evaluate_actions(self, flat_obs_all, goals_all, adj_mask, actions):
        logits, values = self.forward(flat_obs_all, goals_all, adj_mask)
        dist      = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, entropy, values


def goal_reward_shaping(
    state_snapshot,
    goals: np.ndarray,    # (num_zones, GOAL_DIM)
    num_zones: int,
    weight: float = 0.3,
) -> np.ndarray:
    """
    Compute per-intersection intrinsic reward for approaching Manager goals.

    Returns: (num_intersections,) intrinsic reward array.
    The total Worker reward = env_reward + weight * intrinsic_reward.
    """
    intrinsic = np.zeros(len(state_snapshot.intersections), dtype=np.float32)

    for i, s in enumerate(state_snapshot.intersections):
        zone = int(s.id) % num_zones
        if zone >= len(goals):
            continue
        target_tp   = float(goals[zone][0])   # 0-1, normalized target throughput
        target_wait = float(goals[zone][1])   # 0-1, normalized target wait

        actual_tp   = np.clip(s.throughput / 50.0, 0, 1)
        actual_wait = np.clip(s.avg_wait_time / 600.0, 0, 1)

        tp_progress   = actual_tp - target_tp         # positive = exceeding goal
        wait_progress = target_wait - actual_wait     # positive = below target wait

        intrinsic[i] = weight * (0.5 * tp_progress + 0.5 * wait_progress)

    return intrinsic


def flatten_obs_with_goal(obs: dict, goal: np.ndarray) -> np.ndarray:
    """Concatenate local flat obs with goal vector."""
    from rl.agents.marl.ippo_agent import flatten_obs
    local = flatten_obs(obs)
    return np.concatenate([local, goal.astype(np.float32)])
