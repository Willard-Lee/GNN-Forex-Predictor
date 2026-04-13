"""
Dynamic Multi-Edge Graph Construction.

Nodes  = 25 technical indicators
Edges  = weighted combination of 3 correlation types:
  Primary   (40%): 30-day Pearson correlation (|ρ| > 0.45)
  Secondary (40%): 7-day DCC-GARCH proxy (vol-adjusted short corr)
  Tertiary  (20%): Granger causality (p < 0.05, directional)

CRITICAL: Only call on TRAINING data to avoid leakage.
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")


def _pearson_edges(df, feature_cols, window=30, threshold=0.45):
    """30-day Pearson correlation matrix."""
    recent = df[feature_cols].tail(window).dropna()
    if len(recent) < 10:
        return np.zeros((len(feature_cols), len(feature_cols)))
    corr = recent.corr(method="pearson").values
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr, nan=0.0)
    # Apply threshold
    mask = np.abs(corr) < threshold
    corr[mask] = 0.0
    return corr


def _dcc_proxy_edges(df, feature_cols, window=7):
    """
    DCC-GARCH proxy: short-window vol-adjusted correlation.
    Uses standardised returns over the short window.
    """
    recent = df[feature_cols].tail(window).dropna()
    if len(recent) < 5:
        return np.zeros((len(feature_cols), len(feature_cols)))

    # Standardise each column by its own std (vol adjustment)
    stds = recent.std()
    stds = stds.replace(0, 1)
    standardised = (recent - recent.mean()) / stds

    corr = standardised.corr(method="pearson").values
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def _granger_edges(df, feature_cols, max_lag=5, p_threshold=0.05):
    """
    Pairwise Granger causality test.
    Returns a directional matrix: granger[i,j] = 1 if col_i Granger-causes col_j.
    Expensive — O(N² × lags), so we use a simplified F-test.
    """
    n = len(feature_cols)
    granger = np.zeros((n, n))
    data = df[feature_cols].tail(200).dropna()  # use last 200 rows max

    if len(data) < max_lag + 20:
        return granger

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                y = data.iloc[:, j].values
                x = data.iloc[:, i].values

                # Simple F-test: compare AR(lag) model with and without x lags
                T = len(y)
                if T < max_lag + 10:
                    continue

                # Restricted model: y ~ y_lags
                Y = y[max_lag:]
                X_r = np.column_stack([y[max_lag - k - 1 : T - k - 1] for k in range(max_lag)])

                # Unrestricted: y ~ y_lags + x_lags
                X_u = np.column_stack([X_r] + [x[max_lag - k - 1 : T - k - 1] for k in range(max_lag)])

                # OLS residuals
                _, res_r, _, _ = np.linalg.lstsq(X_r, Y, rcond=None)
                _, res_u, _, _ = np.linalg.lstsq(X_u, Y, rcond=None)

                ssr_r = res_r[0] if len(res_r) > 0 else np.sum((Y - X_r @ np.linalg.lstsq(X_r, Y, rcond=None)[0]) ** 2)
                ssr_u = res_u[0] if len(res_u) > 0 else np.sum((Y - X_u @ np.linalg.lstsq(X_u, Y, rcond=None)[0]) ** 2)

                n_obs = len(Y)
                k_diff = max_lag
                k_full = X_u.shape[1]

                if ssr_u > 0 and n_obs > k_full:
                    f_stat = ((ssr_r - ssr_u) / k_diff) / (ssr_u / (n_obs - k_full))
                    p_val = 1 - sp_stats.f.cdf(f_stat, k_diff, n_obs - k_full)
                    if p_val < p_threshold and f_stat > 0:
                        granger[i, j] = 1.0
            except Exception:
                continue

    return granger


def build_feature_graph(df, feature_cols, cfg_graph) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Build multi-edge feature graph from TRAINING data only.

    Returns:
        edge_index:  [2, E] LongTensor
        edge_weight: [E] FloatTensor (composite weight)
        diagnostics: dict with individual correlation matrices
    """
    n_nodes = len(feature_cols)

    print("  📐 Computing Pearson correlations...")
    pearson = _pearson_edges(df, feature_cols, cfg_graph.pearson_window, cfg_graph.pearson_threshold)

    print("  📐 Computing DCC-GARCH proxy...")
    dcc = _dcc_proxy_edges(df, feature_cols, cfg_graph.dcc_window)

    print("  📐 Computing Granger causality...")
    granger = _granger_edges(df, feature_cols, cfg_graph.granger_max_lag, cfg_graph.granger_p_threshold)

    # Composite weight: 40% Pearson + 40% DCC + 20% Granger
    composite = (
        cfg_graph.weight_pearson * np.abs(pearson) +
        cfg_graph.weight_dcc * np.abs(dcc) +
        cfg_graph.weight_granger * granger
    )
    np.fill_diagonal(composite, 0.0)

    # Sparsify: keep top-k per node
    edge_list, weights = [], []
    for i in range(n_nodes):
        row = composite[i]
        top_idx = np.argsort(row)[-(cfg_graph.top_k + 1):]
        for j in top_idx:
            if i != j and row[j] > cfg_graph.min_edge_weight:
                edge_list.append([i, j])
                weights.append(float(row[j]))

    # Fallback
    if not edge_list:
        print("  ⚠️  No edges above threshold — creating fully connected graph")
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_list.append([i, j])
                    weights.append(float(composite[i, j]) if composite[i, j] != 0 else 0.01)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    avg_degree = edge_index.shape[1] / n_nodes
    granger_edges = int(granger.sum())
    print(f"✅ Graph built  |  {n_nodes} nodes, {edge_index.shape[1]} edges  "
          f"|  avg degree: {avg_degree:.1f}  |  Granger edges: {granger_edges}")

    diagnostics = {
        "pearson": pearson,
        "dcc": dcc,
        "granger": granger,
        "composite": composite,
    }
    return edge_index, edge_weight, diagnostics
