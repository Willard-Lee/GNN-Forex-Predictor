"""
Evaluation module — updated for 3-class direction + volatility target.

Metrics:
  Direction: accuracy, F1 (macro), confusion matrix
  Returns:   MAE, directional accuracy (sign match)
  Volatility: quantile loss, calibration (RMSE)
  Portfolio:  Sharpe, Sortino, max drawdown (via backtester)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error,
)
from scipy import stats
from typing import Dict, Optional
import pandas as pd


def predict(model, X, adj=None, edge_weight=None, is_gat=True, device="cpu", batch_size=64):
    """Run inference. Returns dict of numpy arrays."""
    model = model.to(device)
    model.eval()

    ds = TensorDataset(torch.tensor(X, dtype=torch.float))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_dir, all_ret, all_vol, all_att = [], [], [], []
    if adj is not None:
        adj = adj.to(device)

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            if is_gat:
                d, r, v, att = model(batch_x, adj, edge_weight)
                all_att.append(att.cpu().numpy())
            else:
                d, r, v = model(batch_x)

            all_dir.append(torch.softmax(d, dim=1).cpu().numpy())  # (B, 3)
            all_ret.append(r.cpu().numpy())
            all_vol.append(v.cpu().numpy())

    return {
        "dir_probs": np.concatenate(all_dir),         # (N, 3) — softmax
        "dir_pred": np.concatenate(all_dir).argmax(axis=1),  # (N,) — 0/1/2
        "ret_pred": np.concatenate(all_ret).squeeze(),
        "vol_pred": np.concatenate(all_vol).squeeze(),
        "attention": np.concatenate(all_att) if all_att else None,
    }


def compute_metrics(preds, targets):
    """Comprehensive metrics for 3-class direction + regression."""
    dir_pred = preds["dir_pred"]
    dir_true = targets["direction"]
    ret_pred = preds["ret_pred"]
    ret_true = targets["return"]
    vol_pred = preds["vol_pred"]
    vol_true = targets["volatility"]

    m = {}

    # Direction metrics (3-class)
    m["accuracy"] = accuracy_score(dir_true, dir_pred)
    m["f1_macro"] = f1_score(dir_true, dir_pred, average="macro", zero_division=0)
    m["f1_weighted"] = f1_score(dir_true, dir_pred, average="weighted", zero_division=0)

    # Per-class F1
    f1s = f1_score(dir_true, dir_pred, average=None, zero_division=0, labels=[0, 1, 2])
    m["f1_down"] = f1s[0] if len(f1s) > 0 else 0.0
    m["f1_flat"] = f1s[1] if len(f1s) > 1 else 0.0
    m["f1_up"] = f1s[2] if len(f1s) > 2 else 0.0

    # Directional accuracy (sign match for returns)
    sign_pred = np.sign(ret_pred)
    sign_true = np.sign(ret_true)
    m["dir_accuracy_returns"] = (sign_pred == sign_true).mean()

    # Return metrics
    m["ret_rmse"] = np.sqrt(mean_squared_error(ret_true, ret_pred))
    m["ret_mae"] = mean_absolute_error(ret_true, ret_pred)

    # Volatility metrics
    m["vol_rmse"] = np.sqrt(mean_squared_error(vol_true, vol_pred))
    m["vol_mae"] = mean_absolute_error(vol_true, vol_pred)

    return m


def compare_models(gat_preds, base_preds, targets):
    """Side-by-side comparison."""
    gat_m = compute_metrics(gat_preds, targets)
    base_m = compute_metrics(base_preds, targets)

    rows = []
    for key in gat_m:
        gv = gat_m[key]
        bv = base_m[key]
        imp = ((gv - bv) / abs(bv) * 100) if bv != 0 else 0.0
        rows.append({
            "Metric": key,
            "GAT-LSTM": f"{gv:.4f}",
            "Baseline LSTM": f"{bv:.4f}",
            "Improvement %": f"{imp:+.2f}%",
        })
    return pd.DataFrame(rows)


def get_confusion_matrices(gat_preds, base_preds, targets):
    """Return confusion matrices for both models."""
    labels = [0, 1, 2]
    cm_gat = confusion_matrix(targets["direction"], gat_preds["dir_pred"], labels=labels)
    cm_base = confusion_matrix(targets["direction"], base_preds["dir_pred"], labels=labels)
    return cm_gat, cm_base


def significance_test(gat_preds, base_preds, targets):
    """Paired t-test on per-sample errors."""
    ret_true = targets["return"]
    gat_se = (gat_preds["ret_pred"] - ret_true) ** 2
    base_se = (base_preds["ret_pred"] - ret_true) ** 2
    t_ret, p_ret = stats.ttest_rel(gat_se, base_se)

    dir_true = targets["direction"]
    gat_correct = (gat_preds["dir_pred"] == dir_true).astype(float)
    base_correct = (base_preds["dir_pred"] == dir_true).astype(float)
    t_dir, p_dir = stats.ttest_rel(gat_correct, base_correct)

    return {
        "return_t_stat": t_ret, "return_p_value": p_ret,
        "direction_t_stat": t_dir, "direction_p_value": p_dir,
        "significant_return": p_ret < 0.05,
        "significant_direction": p_dir < 0.05,
    }
