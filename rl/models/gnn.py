from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class NeighborAttentionConv(nn.Module):
    """
    Single graph convolution layer using additive attention over neighbors.
    Operates on dense adjacency — suitable for city grids ≤256 nodes.
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4):
        super().__init__()
        assert out_features % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = out_features // num_heads

        self.W_q = nn.Linear(in_features, out_features, bias=False)
        self.W_k = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)
        self.W_o = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(
        self,
        x: torch.Tensor,           # (N, in_features)  OR  (B, N, in_features)
        adj_mask: torch.Tensor,    # (N, N) bool — True where edge exists (incl. self)
    ) -> torch.Tensor:             # same leading dims as x
        # Support both unbatched (N, F) and batched (B, N, F) inputs so the PPO
        # update loop can forward an entire minibatch of timesteps at once instead
        # of calling this once per timestep in a Python for-loop.
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)          # (1, N, F)

        B, N, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        # (B, N, H, D)
        Q = self.W_q(x).view(B, N, H, D)
        K = self.W_k(x).view(B, N, H, D)
        V = self.W_v(x).view(B, N, H, D)

        # Attention scores: (B, H, N, N)
        scores = torch.einsum("bnhd,bmhd->bhnm", Q, K) / (D ** 0.5)
        scores = scores.permute(0, 2, 1, 3)               # (B, N, H, N)

        # adj_mask (N, N) broadcasts over batch and head dimensions
        mask = adj_mask.unsqueeze(0).unsqueeze(2)          # (1, N, 1, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)                      # isolated nodes → 0

        # Aggregate: (B, N, H, D)
        out = torch.einsum("bnhm,bmhd->bnhd", attn, V)
        out = out.reshape(B, N, H * D)
        out = self.W_o(out)
        residual = self.W_q(x)                             # (B, N, out_features)
        result = self.norm(out + residual if out.shape == residual.shape else out)

        return result.squeeze(0) if unbatched else result


class TrafficGNN(nn.Module):
    """
    Two-layer GNN that embeds local intersection observations into a
    communication-aware feature vector, incorporating neighborhood context.

    Input:
        node_features:  (N, node_feat_dim) — local obs for each intersection
        adj_mask:       (N, N) bool — adjacency (neighbors within k hops)

    Output:
        embeddings:     (N, embed_dim) — enriched per-node representation
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 128,
        embed_dim: int  = 64,
        num_heads: int  = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.conv_layers = nn.ModuleList()
        in_dim = hidden_dim
        for i in range(num_layers):
            out_dim = embed_dim if i == num_layers - 1 else hidden_dim
            self.conv_layers.append(NeighborAttentionConv(in_dim, out_dim, num_heads))
            in_dim = out_dim

        self.embed_dim = embed_dim

    def forward(
        self,
        node_features: torch.Tensor,  # (N, node_feat_dim)
        adj_mask: torch.Tensor,       # (N, N) bool
    ) -> torch.Tensor:                # (N, embed_dim)
        h = F.relu(self.input_proj(node_features))
        for conv in self.conv_layers:
            h = F.relu(conv(h, adj_mask))
        return h


def build_adjacency_mask(
    light_node_ids: List[int],
    edges: list,
    k_hops: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a boolean adjacency matrix for traffic-light nodes only.
    Includes self-loops. Optionally expands to k-hop neighborhoods.

    edges: list of EdgeRecord (with .from_node, .to_node attributes).
    """
    n = len(light_node_ids)
    node_to_idx = {nid: i for i, nid in enumerate(light_node_ids)}
    light_set   = set(light_node_ids)

    adj = torch.eye(n, dtype=torch.bool, device=device)

    for edge in edges:
        src, dst = edge.from_node, edge.to_node
        if src in node_to_idx and dst in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[dst]
            adj[i, j] = True
            adj[j, i] = True   # undirected neighborhood for communication

    # Expand to k hops via repeated boolean matrix multiply
    if k_hops > 1:
        reach = adj.clone()
        for _ in range(k_hops - 1):
            reach = torch.logical_or(reach, torch.mm(reach.float(), adj.float()).bool())
        return reach

    return adj
