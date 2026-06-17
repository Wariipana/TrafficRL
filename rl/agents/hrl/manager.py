from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rl.env.data_types import StateSnapshot, EnvConfig


# Number of zone goal dimensions: [target_throughput, target_avg_wait_norm]
GOAL_DIM = 2


@dataclass
class ManagerConfig:
    decision_interval: int = 20   # Worker steps per Manager decision
    hidden_dim: int        = 128
    goal_horizon: float    = 0.95  # γ for Manager's own value estimates
    learning_rate: float   = 1e-4
    clip_eps: float        = 0.2
    ent_coef: float        = 0.005
    vf_coef: float         = 0.5
    n_epochs: int          = 4
    max_zones: int         = 8


class ManagerNetwork(nn.Module):
    """
    High-level policy: maps zone-aggregated state → continuous goal vectors.

    Input:  zone_features  (num_zones, zone_feat_dim)
    Output: goals          (num_zones, GOAL_DIM)  in [0,1]
            value          (num_zones,)
    """

    def __init__(self, zone_feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.zone_encoder = nn.Sequential(
            nn.Linear(zone_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.goal_head  = nn.Linear(hidden_dim, GOAL_DIM)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, zone_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h      = self.zone_encoder(zone_feats)        # (Z, hidden)
        goals  = torch.sigmoid(self.goal_head(h))     # (Z, GOAL_DIM) in [0,1]
        values = self.value_head(h).squeeze(-1)       # (Z,)
        return goals, values


def aggregate_zones(state: StateSnapshot, num_zones: int) -> np.ndarray:
    """
    Aggregate intersection snapshots into per-zone feature vectors.

    Each zone feature: [avg_queue_norm, avg_wait_norm, avg_throughput_norm,
                        max_queue_norm, congestion_indicator]
    Returns: (num_zones, 5)
    """
    zone_feat_dim = 5
    zone_data: Dict[int, list] = {z: [] for z in range(num_zones)}

    for s in state.intersections:
        zone = int(s.id) % num_zones   # simple zone assignment by ID
        zone_data[zone].append(s)

    feats = np.zeros((num_zones, zone_feat_dim), dtype=np.float32)
    for z in range(num_zones):
        snaps = zone_data[z]
        if not snaps:
            continue
        avg_q  = np.mean([np.mean(s.queue_length)  for s in snaps]) / 50.0
        avg_w  = np.mean([s.avg_wait_time           for s in snaps]) / 600.0
        avg_tp = np.mean([s.throughput              for s in snaps]) / 50.0
        max_q  = np.max( [np.max(s.queue_length)   for s in snaps]) / 50.0
        cong   = float(avg_q > 0.6)
        feats[z] = [
            np.clip(avg_q,  0, 1),
            np.clip(avg_w,  0, 1),
            np.clip(avg_tp, 0, 1),
            np.clip(max_q,  0, 1),
            cong,
        ]
    return feats


class HRLManager:
    """
    Stateful Manager that produces goals for Worker agents.

    Usage:
        manager = HRLManager(cfg, num_zones=4)
        # At each step, call maybe_update() — it returns goals only every
        # decision_interval steps, otherwise re-uses the last goals.
        goals = manager.maybe_update(state, step_count)
        # goals: np.ndarray (num_lights,) mapping each light to its zone goal (flat)
    """

    def __init__(self, cfg: ManagerConfig, num_zones: int = 4):
        self.cfg       = cfg
        self.num_zones = num_zones
        self._net      = ManagerNetwork(zone_feat_dim=5, hidden_dim=cfg.hidden_dim)
        self._opt      = torch.optim.Adam(self._net.parameters(), lr=cfg.learning_rate)
        self._last_goals: np.ndarray | None = None   # (num_zones, GOAL_DIM)
        self._last_step  = -1

        # Rollout storage for Manager's own PPO update
        self._rollout: list[dict] = []

    def get_goals(
        self,
        state: StateSnapshot,
        step: int,
        force: bool = False,
    ) -> np.ndarray:
        """
        Returns goal array (num_zones, GOAL_DIM).
        Only recomputes every decision_interval steps.
        """
        if force or self._last_goals is None or (step - self._last_step) >= self.cfg.decision_interval:
            self._last_goals = self._compute_goals(state, step)
            self._last_step  = step
        return self._last_goals

    def goal_for_intersection(self, intersection_id: int) -> np.ndarray:
        """Return the goal vector for a specific intersection (by ID)."""
        if self._last_goals is None:
            return np.zeros(GOAL_DIM, dtype=np.float32)
        zone = intersection_id % self.num_zones
        return self._last_goals[zone].astype(np.float32)

    def record_transition(
        self,
        zone_feats: np.ndarray,
        goals: np.ndarray,
        zone_reward: float,
        done: bool,
    ) -> None:
        self._rollout.append({
            "zone_feats":  zone_feats,
            "goals":       goals,
            "reward":      zone_reward,
            "done":        done,
        })

    def update(self) -> Optional[float]:
        """Run PPO update on accumulated Manager rollout. Returns mean loss or None."""
        if len(self._rollout) < 4:
            return None

        zone_feats_t = torch.tensor(
            np.stack([r["zone_feats"] for r in self._rollout]), dtype=torch.float32)
        goals_t = torch.tensor(
            np.stack([r["goals"]      for r in self._rollout]), dtype=torch.float32)
        rewards  = [r["reward"] for r in self._rollout]
        dones    = [float(r["done"])  for r in self._rollout]

        # Compute discounted returns
        returns = []
        G = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.cfg.goal_horizon * G * (1.0 - d)
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        total_loss = 0.0
        for _ in range(self.cfg.n_epochs):
            pred_goals, values = self._net(zone_feats_t)
            # Manager treats goals as deterministic output; we use MSE as policy loss
            # (simplified FeUdal / HIRO-style: manager trains to match useful goals)
            goal_loss = F.mse_loss(pred_goals, goals_t)
            vf_loss   = F.mse_loss(values, returns_t.unsqueeze(-1).expand_as(values))
            loss = goal_loss + self.cfg.vf_coef * vf_loss

            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
            total_loss += loss.item()

        self._rollout.clear()
        return total_loss / self.cfg.n_epochs

    def save(self, path: str) -> None:
        torch.save(self._net.state_dict(), path)

    def load(self, path: str) -> None:
        self._net.load_state_dict(torch.load(path, map_location="cpu"))

    def _compute_goals(self, state: StateSnapshot, step: int) -> np.ndarray:
        zone_feats = aggregate_zones(state, self.num_zones)  # (Z, 5)
        feats_t    = torch.tensor(zone_feats, dtype=torch.float32)
        with torch.no_grad():
            goals_t, _ = self._net(feats_t)
        return goals_t.numpy()   # (Z, GOAL_DIM)
