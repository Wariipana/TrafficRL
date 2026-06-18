from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl.env.data_types import EnvConfig, MAX_LANES
from rl.models.gnn import TrafficGNN, build_adjacency_mask


# ---- Local observation encoder ----

LOCAL_FEAT_DIM = MAX_LANES * 3 + 1 + 1 + 2 + 3   # vpl + queue + speed + wait + timer + phase_oh + neighbor


class LocalEncoder(nn.Module):
    """Encodes a single intersection's flat observation into a dense vector."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LOCAL_FEAT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, LOCAL_FEAT_DIM) -> (B, out_dim)
        return self.net(x)


def flatten_obs(obs: dict) -> np.ndarray:
    """Convert a single agent's obs dict to a flat numpy array."""
    vpl    = obs["vehicles_per_lane"].flatten()   / 50.0
    ql     = obs["queue_length"].flatten()        / 50.0
    spd    = obs["avg_speed"].flatten()           / 50.0
    wait   = obs["avg_wait_time"].flatten()       / 600.0
    timer  = obs["phase_timer"].flatten()         / 120.0
    phase  = int(obs["current_phase"])
    phase_oh = np.array([float(phase == 0), float(phase == 1)], dtype=np.float32)
    nb     = obs["neighbor_summary"].flatten()
    return np.concatenate([vpl, ql, spd, wait, timer, phase_oh, nb]).astype(np.float32)


# ---- Actor-Critic with GNN communication channel ----

class IPPOActorCritic(nn.Module):
    """
    Shared policy network for Independent PPO across all traffic lights.

    Architecture:
      1. LocalEncoder:  raw obs → local_feat (64d)
      2. TrafficGNN:    all local_feats → gnn_embed (64d), using graph adjacency
      3. Fusion:        [local_feat; gnn_embed] → policy_head / value_head
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
        self.local_encoder = LocalEncoder(out_dim=local_enc_dim)
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
            nn.Linear(actor_hidden, 2),  # logits for Discrete(2)
        )
        self.critic = nn.Sequential(
            nn.Linear(fused_dim, critic_hidden),
            nn.Tanh(),
            nn.Linear(critic_hidden, critic_hidden),
            nn.Tanh(),
            nn.Linear(critic_hidden, 1),
        )

    def forward(
        self,
        flat_obs_all: torch.Tensor,   # (N, LOCAL_FEAT_DIM)  N = num agents
        adj_mask: torch.Tensor,       # (N, N) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns logits (N,2) and values (N,)."""
        local_feat = self.local_encoder(flat_obs_all)        # (N, 64)
        gnn_feat   = self.gnn(local_feat, adj_mask)          # (N, 64)
        fused      = torch.cat([local_feat, gnn_feat], dim=-1)  # (N, 128)
        logits     = self.actor(fused)                       # (N, 2)
        values     = self.critic(fused).squeeze(-1)          # (N,)
        return logits, values

    def get_action(
        self,
        flat_obs_all: torch.Tensor,
        adj_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns actions (N,), log_probs (N,), values (N,)."""
        logits, values = self.forward(flat_obs_all, adj_mask)
        dist   = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    def evaluate_actions(
        self,
        flat_obs_all: torch.Tensor,
        adj_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns log_probs (N,), entropy (N,), values (N,)."""
        logits, values = self.forward(flat_obs_all, adj_mask)
        dist      = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, entropy, values


# ---- IPPO Trainer ----

@dataclass
class IPPOConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    total_timesteps: int    = 2_000_000
    n_steps:         int    = 1024         # steps per rollout per agent
    batch_size:      int    = 256
    n_epochs:        int    = 8
    gamma:           float  = 0.99
    gae_lambda:      float  = 0.95
    clip_eps:        float  = 0.2
    vf_coef:         float  = 0.5
    # Tuned with centralized PPO (same reasoning): 3e-4 was unstable, 1e-4 left the
    # reward flat, 2e-4 is the middle ground. IPPO has no target_kl gate, so the lr
    # is the main knob here. ent_coef kept low so the policy commits.
    ent_coef:        float  = 0.003
    max_grad_norm:   float  = 0.5
    learning_rate:   float  = 2e-4
    k_hops:          int    = 1            # GNN neighborhood hops
    gnn_hidden:      int    = 128
    gnn_embed:       int    = 64
    log_dir:         str    = "rl/runs/ippo"
    save_path:       str    = "rl/models/ippo_gnn"
    seed:            int    = 42
    device:          str    = "cpu"


class RolloutBuffer:
    """Stores (obs, adj, action, log_prob, reward, value, done) tuples."""

    def __init__(self, n_steps: int, n_agents: int, device: torch.device):
        self.n_steps  = n_steps
        self.n_agents = n_agents
        self.device   = device
        self.reset()

    def reset(self) -> None:
        self.obs:       list = []
        self.adj:       list = []
        self.actions:   list = []
        self.log_probs: list = []
        self.rewards:   list = []
        self.values:    list = []
        self.dones:     list = []
        self.ptr = 0

    def add(
        self,
        obs:      torch.Tensor,
        adj:      torch.Tensor,
        actions:  torch.Tensor,
        log_probs:torch.Tensor,
        rewards:  torch.Tensor,
        values:   torch.Tensor,
        dones:    torch.Tensor,
    ) -> None:
        self.obs.append(obs)
        self.adj.append(adj)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.values.append(values)
        self.dones.append(dones)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,  # (N,)
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE advantage estimation. Returns (returns, advantages) each (T, N)."""
        T  = len(self.rewards)
        N  = self.n_agents

        rewards = torch.stack(self.rewards)    # (T, N)
        values  = torch.stack(self.values)     # (T, N)
        dones   = torch.stack(self.dones)      # (T, N)

        advantages = torch.zeros(T, N, device=self.device)
        gae        = torch.zeros(N, device=self.device)

        for t in reversed(range(T)):
            next_val = last_values if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
            gae   = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return returns, advantages


def train_ippo(cfg: IPPOConfig) -> IPPOActorCritic:
    from rl.env.marl_env import MARLTrafficEnv
    from rl.models.gnn import build_adjacency_mask

    os.makedirs(cfg.log_dir,  exist_ok=True)
    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    env   = MARLTrafficEnv(cfg.env)
    obs_d, _ = env.reset(seed=cfg.seed)

    n_agents = env._graph.num_lights
    adj_mask = build_adjacency_mask(
        env._graph.light_node_ids,
        env._graph.edges,
        k_hops=cfg.k_hops,
        device=device,
    )

    model = IPPOActorCritic(
        gnn_hidden=cfg.gnn_hidden,
        gnn_embed=cfg.gnn_embed,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    buffer    = RolloutBuffer(cfg.n_steps, n_agents, device)

    total_steps  = 0
    episode      = 0
    update_count = 0
    # Per-episode reward accumulator so the log shows whether the agent is actually
    # learning (mean reward per agent over the episode) — not just the episode count.
    ep_reward_sum = 0.0
    ep_len        = 0

    while total_steps < cfg.total_timesteps:
        buffer.reset()
        model.eval()

        with torch.no_grad():
            while not buffer.is_full():
                flat_all = torch.tensor(
                    np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(n_agents)]),
                    dtype=torch.float32, device=device,
                )
                actions_t, log_probs_t, values_t = model.get_action(flat_all, adj_mask)

                action_dict = {f"light_{i}": int(actions_t[i]) for i in range(n_agents)}
                obs_d, rews_d, terms_d, trunc_d, _ = env.step(action_dict)

                rewards_t = torch.tensor(
                    [rews_d.get(f"light_{i}", 0.0) for i in range(n_agents)],
                    dtype=torch.float32, device=device,
                )
                dones_t = torch.tensor(
                    [float(terms_d.get(f"light_{i}", False) or trunc_d.get(f"light_{i}", False))
                     for i in range(n_agents)],
                    dtype=torch.float32, device=device,
                )

                buffer.add(flat_all, adj_mask, actions_t, log_probs_t, rewards_t, values_t, dones_t)
                total_steps += 1          # count env steps, not per-agent steps
                ep_reward_sum += float(rewards_t.mean())  # mean reward across agents this step
                ep_len        += 1

                if not env.agents:
                    episode += 1
                    obs_d, _ = env.reset()
                    if episode % 10 == 0:
                        print(f"[IPPO] ep={episode} steps={total_steps} "
                              f"ep_rew_mean={ep_reward_sum:.2f} ep_len={ep_len}")
                    ep_reward_sum = 0.0
                    ep_len        = 0

            # Bootstrap value for GAE
            flat_last = torch.tensor(
                np.stack([flatten_obs(obs_d.get(f"light_{i}", obs_d[list(obs_d.keys())[0]]))
                          for i in range(n_agents)]),
                dtype=torch.float32, device=device,
            )
            _, last_vals = model.forward(flat_last, adj_mask)

        returns, advantages = buffer.compute_returns_and_advantages(
            last_vals, cfg.gamma, cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        model.train()
        T = len(buffer.obs)
        flat_obs_all = torch.stack(buffer.obs)   # (T, N, feat)
        actions_all  = torch.stack(buffer.actions)
        old_lp_all   = torch.stack(buffer.log_probs)

        # The GNN now accepts (B, N, F) batched input — the (N, N) adjacency
        # mask broadcasts over the batch dimension automatically. This replaces
        # the previous per-timestep Python for-loop, which was the main training
        # bottleneck: one forward+backward per minibatch instead of bs individual
        # passes serialised through Python's GIL.
        pg_losses: list = []
        vf_losses: list = []
        ent_vals:  list = []

        bs = max(1, cfg.batch_size // n_agents)
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(T)
            for start in range(0, T, bs):
                idx = perm[start:start + bs]
                # flat_obs_all[idx]: (bs, N, F)  actions_all[idx]: (bs, N)
                new_lp, entropy, vals = model.evaluate_actions(
                    flat_obs_all[idx], adj_mask, actions_all[idx])
                adv   = advantages[idx]                       # (bs, N)
                ratio = torch.exp(new_lp - old_lp_all[idx])  # (bs, N)
                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps),
                ).mean()
                vf_loss = F.mse_loss(vals, returns[idx])
                loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_vals.append(entropy.mean().item())

        update_count += 1
        if update_count % 10 == 0:
            print(f"[IPPO] update={update_count} steps={total_steps} "
                  f"pg_loss={np.mean(pg_losses):.4f} "
                  f"vf_loss={np.mean(vf_losses):.4f} "
                  f"entropy={np.mean(ent_vals):.4f}")

    torch.save(model.state_dict(), cfg.save_path + ".pt")
    env.close()
    print(f"[IPPO] Model saved to {cfg.save_path}.pt")
    return model
