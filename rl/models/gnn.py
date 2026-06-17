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
        x: torch.Tensor,           # (N, in_features)
        adj_mask: torch.Tensor,    # (N, N) bool — True where edge exists (incl. self)
    ) -> torch.Tensor:             # (N, out_features)
        N = x.size(0)
        H = self.num_heads
        D = self.head_dim

        # (N, H, D)
        Q = self.W_q(x).view(N, H, D)
        K = self.W_k(x).view(N, H, D)
        V = self.W_v(x).view(N, H, D)

        # Attention scores: (N, H, N) = Q @ K^T / sqrt(D)
        scores = torch.einsum("nhd,mhd->hnm", Q, K) / (D ** 0.5)   # (H, N, N)
        scores = scores.permute(1, 0, 2)                             # (N, H, N)

        # Mask non-neighbors with -inf before softmax
        mask = adj_mask.unsqueeze(1).expand(-1, H, -1)  # (N, H, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)   # nodes with no neighbors → 0

        # Aggregate: (N, H, D)
        out = torch.einsum("nhm,mhd->nhd", attn, V)
        out = out.reshape(N, H * D)
        out = self.W_o(out)
        return self.norm(out + self.W_q(x) if out.shape == x.shape else out)


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
