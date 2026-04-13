"""
GAT-LSTM Hybrid Model for EUR/USD Forecasting — Upgraded.

Architecture per pipeline spec:
  Spatial  — 2-layer GAT (8→4 heads, 64→128 dim) + LayerNorm + edge dropout
  Temporal — 2-layer LSTM (128 hidden) on full sequence
  Fusion   — concat → shared 128-dim → 3 parallel heads
  Heads    — returns (MSE), volatility (MAE), direction 3-class (CE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# GAT Layer
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """Single-head GAT layer (Veličković et al., 2018)."""

    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        B, N, _ = Wh.size()

        Wh_i = Wh.repeat_interleave(N, dim=1)
        Wh_j = Wh.repeat(1, N, 1)
        a_in = torch.cat([Wh_i, Wh_j], dim=2).view(B, N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_in, self.a).squeeze(3))

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
        neg_inf = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, neg_inf)
        attention = F.softmax(attention, dim=2)

        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime), attention
        return h_prime, attention


class MultiHeadGAT(nn.Module):
    """Multi-head GAT with optional LayerNorm and edge dropout."""

    def __init__(self, in_features, out_features, heads=8, alpha=0.2,
                 concat=True, dropout=0.1, use_layer_norm=True):
        super().__init__()
        self.att_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, alpha, concat=True)
            for _ in range(heads)
        ])
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.use_ln = use_layer_norm
        out_dim = out_features * heads if concat else out_features
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, h, adj):
        outputs, attentions = [], []
        for head in self.att_heads:
            out, att = head(h, adj)
            outputs.append(out)
            attentions.append(att)

        if self.concat:
            x = torch.cat(outputs, dim=-1)
        else:
            x = torch.stack(outputs, dim=0).mean(dim=0)

        x = self.dropout(x)
        if self.use_ln:
            x = self.layer_norm(x)

        att_avg = torch.stack(attentions, dim=0).mean(dim=0)
        return x, att_avg


# ---------------------------------------------------------------------------
# Full GAT-LSTM
# ---------------------------------------------------------------------------

class GATLSTM(nn.Module):
    """
    Hybrid GAT-LSTM — upgraded architecture.

    Path A: 2-layer GAT (8→4 heads) on indicator graph
    Path B: 2-layer LSTM (128 hidden) on sequence
    Heads:  3-class direction (CE), return (MSE), volatility (MAE)
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        seq_len: int = 30,
        gat_hidden: int = 64,
        gat_output: int = 128,
        gat_heads_l1: int = 8,
        gat_heads_l2: int = 4,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        shared_dim: int = 128,
        dropout: float = 0.1,
        edge_dropout: float = 0.15,
        use_layer_norm: bool = True,
        num_direction_classes: int = 3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.edge_dropout = edge_dropout

        # Node feature enrichment: raw + z-score + 3d-slope + 5d-vol = 4 dims
        node_input_dim = 4
        self.node_fc = nn.Linear(node_input_dim, gat_hidden)
        if use_layer_norm:
            self.node_ln = nn.LayerNorm(gat_hidden)
        else:
            self.node_ln = nn.Identity()

        # GAT layer 1: 8 heads, concat → gat_hidden * 8
        self.gat1 = MultiHeadGAT(gat_hidden, gat_hidden, heads=gat_heads_l1,
                                  concat=True, dropout=dropout, use_layer_norm=use_layer_norm)
        gat1_out = gat_hidden * gat_heads_l1

        # GAT layer 2: 4 heads, average → gat_output
        self.gat2 = MultiHeadGAT(gat1_out, gat_output, heads=gat_heads_l2,
                                  concat=False, dropout=dropout, use_layer_norm=use_layer_norm)

        gat_flat = num_nodes * gat_output

        # LSTM temporal block
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        if use_layer_norm:
            self.lstm_ln = nn.LayerNorm(lstm_hidden)
        else:
            self.lstm_ln = nn.Identity()

        fusion_dim = gat_flat + lstm_hidden

        # Shared embedding
        self.fc_shared = nn.Sequential(
            nn.Linear(fusion_dim, shared_dim),
            nn.LayerNorm(shared_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 3 parallel prediction heads
        self.head_direction = nn.Linear(shared_dim, num_direction_classes)  # 3-class
        self.head_return = nn.Linear(shared_dim, 1)
        self.head_volatility = nn.Linear(shared_dim, 1)

    def _build_node_features(self, x_seq):
        """Augment each node: [raw, z-score, 3d-slope, 5d-vol]."""
        B, T, n_feat = x_seq.shape
        N = self.num_nodes

        latest = x_seq[:, -1, :N]  # (B, N)

        # z-score over last 30 steps
        window = x_seq[:, :, :N]  # (B, T, N)
        mean = window.mean(dim=1)
        std = window.std(dim=1).clamp(min=1e-8)
        z_score = (latest - mean) / std

        # 3-day slope
        if T >= 3:
            slope_3d = latest - x_seq[:, -3, :N]
        else:
            slope_3d = torch.zeros_like(latest)

        # 5-day volatility
        if T >= 5:
            vol_5d = x_seq[:, -5:, :N].std(dim=1)
        else:
            vol_5d = window.std(dim=1)

        node_feat = torch.stack([latest, z_score, slope_3d, vol_5d], dim=-1)  # (B, N, 4)
        return node_feat

    def forward(self, x_seq, adj, edge_weight=None):
        """
        Returns:
            dir_logits:  (B, 3) — 3-class logits
            ret_pred:    (B, 1)
            vol_pred:    (B, 1)
            att_weights: (B, N, N)
        """
        B = x_seq.shape[0]

        # ── Path A: GAT ──
        node_feat = self._build_node_features(x_seq)  # (B, N, 4)
        node_feat = torch.relu(self.node_fc(node_feat))
        node_feat = self.node_ln(node_feat)

        # Edge dropout during training
        if self.training and self.edge_dropout > 0:
            mask = (torch.rand_like(adj.float()) > self.edge_dropout).float()
            adj_drop = adj * mask
        else:
            adj_drop = adj

        gat_out, _ = self.gat1(node_feat, adj_drop)
        gat_out, att = self.gat2(gat_out, adj_drop)
        gat_flat = gat_out.reshape(B, -1)

        # ── Path B: LSTM ──
        lstm_out, _ = self.lstm(x_seq)
        lstm_last = self.lstm_ln(lstm_out[:, -1, :])

        # ── Fusion ──
        fused = torch.cat([gat_flat, lstm_last], dim=-1)
        shared = self.fc_shared(fused)

        dir_logits = self.head_direction(shared)
        ret_pred = self.head_return(shared)
        vol_pred = self.head_volatility(shared)

        return dir_logits, ret_pred, vol_pred, att
