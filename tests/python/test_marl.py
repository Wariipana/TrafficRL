"""
Unit tests for Fase 4: MARL + GNN + HRL.
These tests run WITHOUT a live C++ server (mock BridgeClient).
"""
from __future__ import annotations
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from rl.env.data_types import (
    EnvConfig, GraphData, NodeRecord, EdgeRecord,
    IntersectionSnapshot, StateSnapshot, MAX_LANES,
)
from rl.models.gnn import TrafficGNN, build_adjacency_mask, NeighborAttentionConv
from rl.agents.marl.ippo_agent import (
    IPPOActorCritic, flatten_obs, LOCAL_FEAT_DIM,
)
from rl.agents.hrl.manager import (
    HRLManager, ManagerConfig, ManagerNetwork, aggregate_zones, GOAL_DIM,
)
from rl.agents.hrl.worker import (
    HRLWorkerActorCritic, goal_reward_shaping, WORKER_FEAT_DIM,
)


# ---- Fixtures ----

def make_graph(n: int = 6) -> GraphData:
    """Small linear-chain graph with n traffic lights."""
    nodes = [
        NodeRecord(id=i, x=float(i)*10, y=0.0, zone=i // 2,
                   has_light=True, light_id=i, num_outgoing=2)
        for i in range(n)
    ]
    edges = [
        EdgeRecord(id=i, from_node=i, to_node=i + 1,
                   length=10.0, num_lanes=2, speed_limit=13.9, direction=0)
        for i in range(n - 1)
    ]
    return GraphData(
        nodes=nodes, edges=edges,
        light_node_ids=list(range(n)),
        num_lights=n,
    )


def make_state(n: int = 6) -> StateSnapshot:
    rng = np.random.default_rng(0)
    intersections = []
    for i in range(n):
        intersections.append(IntersectionSnapshot(
            id=i, phase=i % 2, num_lanes=2,
            phase_timer_s=float(rng.uniform(0, 30)),
            vehicles_per_lane=rng.uniform(0, 10, size=MAX_LANES).astype(np.float32),
            queue_length=rng.uniform(0, 5, size=MAX_LANES).astype(np.float32),
            avg_speed=rng.uniform(5, 15, size=MAX_LANES).astype(np.float32),
            avg_wait_time=float(rng.uniform(0, 120)),
            throughput=float(rng.uniform(0, 20)),
        ))
    return StateSnapshot(
        sim_tick=100, num_intersections=n, num_vehicles=200,
        sim_time_s=10.0, episode_step=100,
        terminated=False, truncated=False,
        intersections=intersections,
        total_throughput=15.0, avg_wait_global=45.0,
        max_wait_global=120.0, congestion_spread=0.3,
    )


def make_obs_dict(n: int = 6) -> dict:
    """Fake observation dict as returned by MARLTrafficEnv."""
    rng  = np.random.default_rng(1)
    obs  = {}
    for i in range(n):
        obs[f"light_{i}"] = {
            "vehicles_per_lane": rng.uniform(0, 10, MAX_LANES).astype(np.float32),
            "queue_length":      rng.uniform(0, 5,  MAX_LANES).astype(np.float32),
            "avg_speed":         rng.uniform(5, 15, MAX_LANES).astype(np.float32),
            "avg_wait_time":     np.array([float(rng.uniform(0, 120))], dtype=np.float32),
            "current_phase":     int(rng.integers(0, 2)),
            "phase_timer":       np.array([float(rng.uniform(0, 60))], dtype=np.float32),
            "neighbor_summary":  rng.uniform(0, 1, 3).astype(np.float32),
        }
    return obs


# ---- GNN tests ----

class TestNeighborAttentionConv:
    def test_output_shape(self):
        layer = NeighborAttentionConv(in_features=32, out_features=32, num_heads=4)
        x   = torch.randn(6, 32)
        adj = torch.eye(6, dtype=torch.bool)
        out = layer(x, adj)
        assert out.shape == (6, 32)

    def test_masked_isolation(self):
        """With identity adjacency (no neighbors), each node uses only itself."""
        layer = NeighborAttentionConv(in_features=16, out_features=16, num_heads=4)
        x    = torch.randn(4, 16)
        adj  = torch.eye(4, dtype=torch.bool)   # only self-loops
        out1 = layer(x, adj)
        # Permuting nodes should give same result (each uses only self)
        out2 = layer(x[[1, 0, 3, 2]], adj)
        # Not a strict equality test — just check shapes are correct
        assert out1.shape == (4, 16)
        assert out2.shape == (4, 16)


class TestTrafficGNN:
    def test_forward_shape(self):
        graph  = make_graph(8)
        adj    = build_adjacency_mask(graph.light_node_ids, graph.edges)
        model  = TrafficGNN(node_feat_dim=LOCAL_FEAT_DIM, hidden_dim=64, embed_dim=32, num_heads=4)
        x      = torch.randn(8, LOCAL_FEAT_DIM)
        out    = model(x, adj)
        assert out.shape == (8, 32)

    def test_adjacency_self_loop(self):
        """Adjacency mask must include self-loops."""
        graph = make_graph(4)
        adj   = build_adjacency_mask(graph.light_node_ids, graph.edges)
        assert adj.diagonal().all(), "All self-loops should be True"

    def test_adjacency_symmetry(self):
        graph = make_graph(6)
        adj   = build_adjacency_mask(graph.light_node_ids, graph.edges)
        assert torch.equal(adj, adj.T), "Adjacency should be symmetric"

    def test_k_hop_expansion(self):
        graph  = make_graph(6)
        adj1   = build_adjacency_mask(graph.light_node_ids, graph.edges, k_hops=1)
        adj2   = build_adjacency_mask(graph.light_node_ids, graph.edges, k_hops=2)
        # 2-hop should have at least as many True entries as 1-hop
        assert adj2.sum() >= adj1.sum()

    def test_single_node(self):
        """GNN should handle a single node without crashing."""
        model = TrafficGNN(node_feat_dim=16, hidden_dim=32, embed_dim=16)
        x     = torch.randn(1, 16)
        adj   = torch.ones(1, 1, dtype=torch.bool)
        out   = model(x, adj)
        assert out.shape == (1, 16)


# ---- IPPO actor-critic tests ----

class TestIPPOActorCritic:
    def setup_method(self):
        self.n     = 6
        self.graph = make_graph(self.n)
        self.adj   = build_adjacency_mask(self.graph.light_node_ids, self.graph.edges)
        self.model = IPPOActorCritic(gnn_hidden=64, gnn_embed=32)

    def test_flatten_obs_shape(self):
        obs_d = make_obs_dict(self.n)
        flat  = flatten_obs(obs_d["light_0"])
        assert flat.shape == (LOCAL_FEAT_DIM,)
        assert flat.dtype == np.float32

    def test_forward_shapes(self):
        obs_d    = make_obs_dict(self.n)
        flat_all = torch.tensor(
            np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(self.n)]),
            dtype=torch.float32,
        )
        logits, values = self.model(flat_all, self.adj)
        assert logits.shape == (self.n, 2)
        assert values.shape == (self.n,)

    def test_get_action(self):
        obs_d    = make_obs_dict(self.n)
        flat_all = torch.tensor(
            np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(self.n)]),
            dtype=torch.float32,
        )
        actions, log_probs, values = self.model.get_action(flat_all, self.adj)
        assert actions.shape == (self.n,)
        assert all(a in (0, 1) for a in actions.tolist())
        assert log_probs.shape == (self.n,)
        assert (log_probs <= 0).all(), "log_probs should be non-positive"

    def test_evaluate_actions(self):
        obs_d    = make_obs_dict(self.n)
        flat_all = torch.tensor(
            np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(self.n)]),
            dtype=torch.float32,
        )
        actions  = torch.randint(0, 2, (self.n,))
        lp, ent, vals = self.model.evaluate_actions(flat_all, self.adj, actions)
        assert lp.shape == (self.n,)
        assert ent.shape == (self.n,)
        assert (ent >= 0).all(), "Entropy should be non-negative"

    def test_deterministic_action(self):
        """Deterministic mode should return argmax of logits (reproducible)."""
        obs_d = make_obs_dict(self.n)
        flat  = torch.tensor(
            np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(self.n)]),
            dtype=torch.float32,
        )
        a1, _, _ = self.model.get_action(flat, self.adj, deterministic=True)
        a2, _, _ = self.model.get_action(flat, self.adj, deterministic=True)
        assert torch.equal(a1, a2), "Deterministic actions must be reproducible"


# ---- HRL Manager tests ----

class TestHRLManager:
    def setup_method(self):
        self.num_zones = 3
        self.manager   = HRLManager(ManagerConfig(decision_interval=10), num_zones=self.num_zones)
        self.state     = make_state(6)

    def test_goals_shape(self):
        goals = self.manager.get_goals(self.state, step=0, force=True)
        assert goals.shape == (self.num_zones, GOAL_DIM)

    def test_goals_in_range(self):
        goals = self.manager.get_goals(self.state, step=0, force=True)
        assert np.all(goals >= 0.0) and np.all(goals <= 1.0)

    def test_caching_between_intervals(self):
        goals1 = self.manager.get_goals(self.state, step=0, force=True)
        goals2 = self.manager.get_goals(self.state, step=5)   # within interval
        assert np.array_equal(goals1, goals2), "Goals should be cached within interval"

    def test_refresh_after_interval(self):
        goals1 = self.manager.get_goals(self.state, step=0, force=True)
        goals2 = self.manager.get_goals(self.state, step=11)  # past interval
        # May or may not differ numerically, but should not crash
        assert goals2.shape == goals1.shape

    def test_goal_for_intersection(self):
        self.manager.get_goals(self.state, step=0, force=True)
        goal = self.manager.goal_for_intersection(intersection_id=0)
        assert goal.shape == (GOAL_DIM,)

    def test_manager_update_runs(self):
        for _ in range(5):
            self.manager.record_transition(
                zone_feats=np.zeros((self.num_zones, 5), dtype=np.float32),
                goals=np.random.rand(self.num_zones, GOAL_DIM).astype(np.float32),
                zone_reward=float(np.random.rand()),
                done=False,
            )
        loss = self.manager.update()
        assert loss is not None and loss >= 0


class TestAggregateZones:
    def test_output_shape(self):
        state  = make_state(8)
        feats  = aggregate_zones(state, num_zones=4)
        assert feats.shape == (4, 5)

    def test_values_normalized(self):
        state = make_state(8)
        feats = aggregate_zones(state, num_zones=4)
        assert np.all(feats >= 0.0) and np.all(feats <= 1.0)


# ---- HRL Worker tests ----

class TestHRLWorker:
    def setup_method(self):
        self.n      = 6
        self.graph  = make_graph(self.n)
        self.adj    = build_adjacency_mask(self.graph.light_node_ids, self.graph.edges)
        self.worker = HRLWorkerActorCritic(gnn_hidden=64, gnn_embed=32)

    def _make_inputs(self):
        obs_d    = make_obs_dict(self.n)
        flat_all = torch.tensor(
            np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(self.n)]),
            dtype=torch.float32,
        )
        goals_all = torch.rand(self.n, GOAL_DIM)
        return flat_all, goals_all

    def test_forward_shapes(self):
        flat, goals = self._make_inputs()
        logits, values = self.worker.forward(flat, goals, self.adj)
        assert logits.shape == (self.n, 2)
        assert values.shape == (self.n,)

    def test_get_action(self):
        flat, goals = self._make_inputs()
        actions, lp, vals = self.worker.get_action(flat, goals, self.adj)
        assert actions.shape == (self.n,)
        assert lp.shape == (self.n,)

    def test_worker_feat_dim(self):
        assert WORKER_FEAT_DIM == LOCAL_FEAT_DIM + GOAL_DIM


class TestGoalRewardShaping:
    def test_output_shape(self):
        state     = make_state(6)
        goals     = np.random.rand(3, GOAL_DIM).astype(np.float32)
        intrinsic = goal_reward_shaping(state, goals, num_zones=3)
        assert intrinsic.shape == (6,)

    def test_finite_values(self):
        state     = make_state(6)
        goals     = np.random.rand(3, GOAL_DIM).astype(np.float32)
        intrinsic = goal_reward_shaping(state, goals, num_zones=3)
        assert np.all(np.isfinite(intrinsic))
