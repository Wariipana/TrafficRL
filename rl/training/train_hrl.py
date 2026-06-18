"""
HRL training: Manager (zone-level) + Worker (intersection-level IPPO + GNN).

Usage:
    python3 -m rl.training.train_hrl --config config/city_small.yaml --steps 500000
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import torch

from rl.env.data_types import EnvConfig
from rl.env.marl_env import MARLTrafficEnv
from rl.models.gnn import build_adjacency_mask
from rl.agents.marl.ippo_agent import flatten_obs, LOCAL_FEAT_DIM, IPPOConfig
from rl.agents.hrl.manager import HRLManager, ManagerConfig, aggregate_zones, GOAL_DIM
from rl.agents.hrl.worker import HRLWorkerActorCritic, goal_reward_shaping, flatten_obs_with_goal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="config/city_small.yaml")
    p.add_argument("--steps",   type=int, default=500_000)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--device",  default="cpu")
    p.add_argument("--save-dir", default="rl/models/hrl")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    env_cfg = EnvConfig.from_yaml(args.config)
    env     = MARLTrafficEnv(env_cfg)
    obs_d, _ = env.reset(seed=args.seed)

    n_agents  = env._graph.num_lights
    num_zones = min(8, max(1, n_agents // 4))   # ~4 lights per zone

    adj_mask = build_adjacency_mask(
        env._graph.light_node_ids,
        env._graph.edges,
        k_hops=1,
        device=device,
    )

    worker  = HRLWorkerActorCritic().to(device)
    manager = HRLManager(ManagerConfig(), num_zones=num_zones)

    w_opt   = torch.optim.Adam(worker.parameters(), lr=2e-4)  # tuned with PPO/IPPO

    os.makedirs(args.save_dir, exist_ok=True)
    total_steps  = 0
    episode      = 0
    w_rollout: list[dict] = []
    # Per-episode reward accumulators so the log shows whether learning progresses:
    # the Worker's total reward (env + intrinsic goal shaping) and the env-only
    # reward (comparable to the other algorithms' signal).
    ep_worker_rew = 0.0
    ep_env_rew    = 0.0
    ep_len        = 0

    print(f"[HRL] agents={n_agents}  zones={num_zones}  device={device}")

    while total_steps < args.steps:
        state = env._last_state

        # Manager produces goals (every decision_interval steps)
        goals_np = manager.get_goals(state, total_steps)      # (Z, GOAL_DIM)
        zone_feats = aggregate_zones(state, num_zones)

        # Build Worker inputs
        flat_local = np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(n_agents)])
        goals_per_agent = np.stack([
            goals_np[int(env._graph.light_node_ids[i]) % num_zones]
            for i in range(n_agents)
        ])   # (N, GOAL_DIM)

        flat_t     = torch.tensor(flat_local,      dtype=torch.float32, device=device)
        goals_t    = torch.tensor(goals_per_agent, dtype=torch.float32, device=device)

        with torch.no_grad():
            actions_t, log_probs_t, values_t = worker.get_action(flat_t, goals_t, adj_mask)

        action_dict = {f"light_{i}": int(actions_t[i]) for i in range(n_agents)}
        obs_d, rews_d, terms_d, trunc_d, _ = env.step(action_dict)
        total_steps += n_agents

        # Intrinsic reward shaping
        intrinsic = goal_reward_shaping(env._last_state, goals_np, num_zones)
        rewards_np = np.array([
            rews_d.get(f"light_{i}", 0.0) + intrinsic[i]
            for i in range(n_agents)
        ], dtype=np.float32)

        ep_worker_rew += float(rewards_np.mean())                  # env + intrinsic
        ep_env_rew    += float(np.mean(list(rews_d.values())))     # env only
        ep_len        += 1

        dones_np = np.array([
            float(terms_d.get(f"light_{i}", False) or trunc_d.get(f"light_{i}", False))
            for i in range(n_agents)
        ], dtype=np.float32)

        w_rollout.append({
            "flat":     flat_t,
            "goals":    goals_t,
            "actions":  actions_t,
            "log_probs": log_probs_t,
            "rewards":  torch.tensor(rewards_np, device=device),
            "values":   values_t,
            "dones":    torch.tensor(dones_np, device=device),
        })

        # Manager transition
        manager.record_transition(
            zone_feats=zone_feats,
            goals=goals_np,
            zone_reward=float(np.mean(list(rews_d.values()))),
            done=bool(np.any(dones_np)),
        )

        if not env.agents:
            episode += 1
            obs_d, _ = env.reset()
            # Update the Manager every episode (it was previously updated only on
            # every 5th episode, wasting 4/5 of the collected transitions).
            m_loss = manager.update()
            if episode % 5 == 0:
                mloss_s = f"manager_loss={m_loss:.4f}" if m_loss is not None else "manager_loss=n/a"
                print(f"[HRL] ep={episode} steps={total_steps} "
                      f"ep_rew_mean={ep_worker_rew:.2f} env_rew={ep_env_rew:.2f} "
                      f"ep_len={ep_len} {mloss_s}")
            ep_worker_rew = 0.0
            ep_env_rew    = 0.0
            ep_len        = 0

        # Worker PPO update every 1024 steps (per agent)
        if len(w_rollout) >= 1024 // n_agents:
            _update_worker(worker, w_opt, w_rollout, adj_mask, device)
            w_rollout.clear()

    worker_path  = os.path.join(args.save_dir, "worker.pt")
    manager_path = os.path.join(args.save_dir, "manager.pt")
    torch.save(worker.state_dict(), worker_path)
    manager.save(manager_path)
    env.close()
    print(f"[HRL] Saved worker → {worker_path}")
    print(f"[HRL] Saved manager → {manager_path}")


def _update_worker(
    worker: HRLWorkerActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: list[dict],
    adj_mask: torch.Tensor,
    device: torch.device,
    gamma:     float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps:  float = 0.2,
    vf_coef:   float = 0.5,
    ent_coef:  float = 0.003,   # lowered like PPO/IPPO so the policy commits
    n_epochs:  int   = 4,
) -> None:
    import torch.nn.functional as F

    T = len(rollout)
    rewards = torch.stack([r["rewards"] for r in rollout])    # (T, N)
    values  = torch.stack([r["values"]  for r in rollout])    # (T, N)
    dones   = torch.stack([r["dones"]   for r in rollout])    # (T, N)

    # GAE
    advantages = torch.zeros_like(rewards, device=device)
    gae        = torch.zeros(rewards.shape[1], device=device)
    for t in reversed(range(T)):
        nv    = values[t + 1] if t < T - 1 else torch.zeros_like(values[0])
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        gae   = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns     = advantages + values
    advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    flat_all    = torch.stack([r["flat"]      for r in rollout])   # (T, N, feat)
    goals_all   = torch.stack([r["goals"]     for r in rollout])   # (T, N, GOAL_DIM)
    actions_all = torch.stack([r["actions"]   for r in rollout])   # (T, N)
    old_lp_all  = torch.stack([r["log_probs"] for r in rollout])   # (T, N)

    # GNN now accepts (B, N, F) — no per-timestep Python loop needed.
    worker.train()
    bs = max(1, T // 4)
    for _ in range(n_epochs):
        perm = torch.randperm(T)
        for start in range(0, T, bs):
            idx = perm[start:start + bs]
            new_lp, entropy, vals = worker.evaluate_actions(
                flat_all[idx], goals_all[idx], adj_mask, actions_all[idx])
            adv   = advantages[idx]                        # (bs, N)
            ratio = torch.exp(new_lp - old_lp_all[idx])   # (bs, N)
            pg_loss = torch.max(
                -adv * ratio,
                -adv * ratio.clamp(1 - clip_eps, 1 + clip_eps),
            ).mean()
            vf_loss = F.mse_loss(vals, returns[idx])
            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(worker.parameters(), 0.5)
            optimizer.step()

    worker.eval()


if __name__ == "__main__":
    main()
