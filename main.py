"""
Main entry point — GAT-LSTM EUR/USD Forecasting System (v2).

Usage:
    python main.py train
    python main.py eval
    python main.py backtest
    python main.py all
"""

import sys, os, json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from configs.config import Config
from utils.data_pipeline import run_pipeline
from utils.graph_builder import build_feature_graph
from utils.trainer import train_model
from utils.evaluator import predict, compute_metrics, compare_models, significance_test, get_confusion_matrices
from utils.backtester import Backtester, format_backtest_table
from models.gat_lstm import GATLSTM
from baselines.lstm_baseline import BaselineLSTM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_adjacency(edge_index, num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k].item(), edge_index[1, k].item()
        adj[i, j] = 1.0
    return adj


def save_plots(hist_gat, hist_base, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(hist_gat["train_loss"], label="GAT-LSTM Train", color="#2563eb")
    ax.plot(hist_gat["val_loss"], label="GAT-LSTM Val", color="#2563eb", ls="--")
    ax.plot(hist_base["train_loss"], label="LSTM Train", color="#dc2626")
    ax.plot(hist_base["val_loss"], label="LSTM Val", color="#dc2626", ls="--")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training & Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(hist_gat["train_acc"], label="GAT-LSTM Train", color="#2563eb")
    ax.plot(hist_gat["val_acc"], label="GAT-LSTM Val", color="#2563eb", ls="--")
    ax.plot(hist_base["train_acc"], label="LSTM Train", color="#dc2626")
    ax.plot(hist_base["val_acc"], label="LSTM Val", color="#dc2626", ls="--")
    ax.axhline(y=1/3, color="gray", ls=":", label="Random (33%)")
    ax.set(xlabel="Epoch", ylabel="Accuracy", title="3-Class Direction Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved training_curves.png")


def save_confusion_matrices(cm_gat, cm_base, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    labels = ["Down", "Flat", "Up"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in zip(axes, [cm_gat, cm_base], ["GAT-LSTM", "Baseline LSTM"]):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                    yticklabels=labels, ax=ax, square=True)
        ax.set(xlabel="Predicted", ylabel="Actual", title=f"{title} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved confusion_matrices.png")


def save_backtest_plots(results_list, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    ax = axes[0]
    for r, c in zip(results_list, colors):
        ax.plot(r["equity_curve"], label=f"{r['name']} (Sharpe: {r['sharpe_ratio']:.2f})", color=c, lw=1.5)
    ax.axhline(y=100_000, color="gray", ls=":", alpha=0.5)
    ax.set(ylabel="Portfolio Value ($)", title="Equity Curves — Test Period")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for r, c in zip(results_list, colors):
        eq = r["equity_curve"]
        rm = np.maximum.accumulate(eq)
        dd = (rm - eq) / rm * 100
        ax.fill_between(range(len(dd)), dd, alpha=0.2, color=c)
        ax.plot(dd, label=r["name"], color=c, lw=1)
    ax.set(ylabel="Drawdown (%)", xlabel="Trading Days", title="Drawdown")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{out_dir}/backtest_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved backtest_results.png")


def save_attention_heatmap(attention, feature_cols, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    if attention is None:
        return
    avg = attention.mean(axis=0)
    if avg.ndim > 2:
        avg = avg.mean(axis=0)
    n = min(len(feature_cols), avg.shape[0])
    avg = avg[:n, :n]

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(avg, xticklabels=feature_cols[:n], yticklabels=feature_cols[:n],
                cmap="YlOrRd", annot=True, fmt=".2f", ax=ax, square=True,
                annot_kws={"size": 6})
    ax.set_title("GAT Attention Weights (avg over test set)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/attention_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved attention_heatmap.png")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _make_gat_model(cfg, num_nodes):
    return GATLSTM(
        num_nodes=num_nodes,
        num_features=num_nodes,
        seq_len=cfg.data.sequence_length,
        gat_hidden=cfg.model.gat_hidden,
        gat_output=cfg.model.gat_output,
        gat_heads_l1=cfg.model.gat_heads_layer1,
        gat_heads_l2=cfg.model.gat_heads_layer2,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        shared_dim=cfg.model.shared_dim,
        dropout=cfg.model.dropout,
        edge_dropout=cfg.model.edge_dropout,
        use_layer_norm=cfg.model.use_layer_norm,
        num_direction_classes=cfg.model.num_direction_classes,
    )


def _make_base_model(cfg, num_nodes):
    return BaselineLSTM(
        num_features=num_nodes,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        shared_dim=cfg.model.shared_dim,
        dropout=cfg.model.dropout,
        use_layer_norm=cfg.model.use_layer_norm,
        num_direction_classes=cfg.model.num_direction_classes,
    )


def cmd_train(cfg, data):
    print("\n" + "=" * 70)
    print("🏋️  TRAINING")
    print("=" * 70)

    feat = cfg.data.feature_nodes
    num_nodes = len(feat)
    device = cfg.device

    print("\n📐 Building multi-edge feature graph (train data only)...")
    edge_index, edge_weight, diag = build_feature_graph(data["train_df"], feat, cfg.graph)
    adj = build_adjacency(edge_index, num_nodes)

    # GAT-LSTM
    gat_model = _make_gat_model(cfg, num_nodes)
    gat_result = train_model(gat_model, data["train_X"], data["train_y"],
                             data["val_X"], data["val_y"], cfg,
                             adj=adj, edge_weight=edge_weight, is_gat=True, device=device)

    # Baseline LSTM
    base_model = _make_base_model(cfg, num_nodes)
    base_result = train_model(base_model, data["train_X"], data["train_y"],
                              data["val_X"], data["val_y"], cfg,
                              is_gat=False, device=device)

    os.makedirs("outputs/models", exist_ok=True)
    torch.save(gat_result["model"].state_dict(), "outputs/models/gat_lstm.pth")
    torch.save(base_result["model"].state_dict(), "outputs/models/lstm_baseline.pth")
    torch.save({"edge_index": edge_index, "edge_weight": edge_weight, "adj": adj}, "outputs/models/graph.pth")
    print("\n✅ Models saved to outputs/models/")

    save_plots(gat_result["history"], base_result["history"])
    return gat_result, base_result, adj, edge_index, edge_weight


def cmd_eval(cfg, data, gat_model, base_model, adj, edge_index, edge_weight):
    print("\n" + "=" * 70)
    print("📊 EVALUATION ON TEST SET")
    print("=" * 70)

    device = cfg.device
    feat = cfg.data.feature_nodes

    gat_preds = predict(gat_model, data["test_X"], adj=adj, edge_weight=edge_weight,
                        is_gat=True, device=device)
    base_preds = predict(base_model, data["test_X"], is_gat=False, device=device)
    targets = data["test_y"]

    comp = compare_models(gat_preds, base_preds, targets)
    print("\n" + comp.to_string(index=False))

    # Confusion matrices
    cm_gat, cm_base = get_confusion_matrices(gat_preds, base_preds, targets)
    save_confusion_matrices(cm_gat, cm_base)
    print(f"\n  GAT-LSTM confusion matrix:\n{cm_gat}")
    print(f"\n  Baseline confusion matrix:\n{cm_base}")

    sig = significance_test(gat_preds, base_preds, targets)
    print(f"\n📈 Statistical Significance:")
    print(f"   Return RMSE   — t={sig['return_t_stat']:.3f}, p={sig['return_p_value']:.4f} "
          f"{'✅' if sig['significant_return'] else '⚠️'}")
    print(f"   Direction Acc — t={sig['direction_t_stat']:.3f}, p={sig['direction_p_value']:.4f} "
          f"{'✅' if sig['significant_direction'] else '⚠️'}")

    save_attention_heatmap(gat_preds.get("attention"), feat)

    os.makedirs("outputs", exist_ok=True)
    metrics_out = {
        "gat_lstm": compute_metrics(gat_preds, targets),
        "baseline_lstm": compute_metrics(base_preds, targets),
        "significance": sig,
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\n✅ Metrics saved to outputs/metrics.json")

    return gat_preds, base_preds


def cmd_backtest(cfg, data, gat_preds, base_preds):
    print("\n" + "=" * 70)
    print("💰 BACKTESTING")
    print("=" * 70)

    bt = Backtester(cfg.backtest)
    test_df = data["test_df"]
    sl = cfg.data.sequence_length

    results = [
        bt.run_gat_strategy(test_df, gat_preds["dir_probs"], gat_preds["ret_pred"], seq_offset=sl),
        bt.run_lstm_strategy(test_df, base_preds["dir_probs"], base_preds["ret_pred"], seq_offset=sl),
        bt.run_ma_crossover(test_df),
        bt.run_buy_and_hold(test_df),
    ]

    table = format_backtest_table(results)
    print("\n" + table.to_string(index=False))
    save_backtest_plots(results)

    summary = []
    for r in results:
        s = {k: v for k, v in r.items() if k not in ("equity_curve", "trades")}
        s["num_trades_detail"] = {
            "total": r["num_trades"],
            "wins": sum(1 for t in r["trades"] if t.pnl > 0),
            "losses": sum(1 for t in r["trades"] if t.pnl <= 0),
        }
        summary.append(s)
    with open("outputs/backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✅ Backtest results saved")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    valid = {"train", "eval", "backtest", "all"}
    if command not in valid:
        print(f"Usage: python main.py [{' | '.join(valid)}]")
        sys.exit(1)

    cfg = Config()
    cfg.device = get_device()
    set_seed(cfg.train.seed)

    print("=" * 70)
    print("  GAT-LSTM EUR/USD Forecasting System (v2)")
    print(f"  25 Indicators | Multi-Edge Graph | 3-Class Direction")
    print(f"  Device: {cfg.device}  |  Seed: {cfg.train.seed}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    if not os.path.exists(cfg.data.csv_path):
        print(f"\n❌ Data file not found: {cfg.data.csv_path}")
        print(f"   Place your EURUSD_daily.csv in the data/ folder.")
        sys.exit(1)

    print("\n📦 Running data pipeline...")
    data = run_pipeline(cfg)

    num_nodes = len(cfg.data.feature_nodes)

    if command in ("train", "all"):
        gat_res, base_res, adj, ei, ew = cmd_train(cfg, data)
        gat_model = gat_res["model"]
        base_model = base_res["model"]
    else:
        graph_data = torch.load("outputs/models/graph.pth", weights_only=False)
        adj, ei, ew = graph_data["adj"], graph_data["edge_index"], graph_data["edge_weight"]
        gat_model = _make_gat_model(cfg, num_nodes)
        gat_model.load_state_dict(torch.load("outputs/models/gat_lstm.pth", weights_only=False))
        base_model = _make_base_model(cfg, num_nodes)
        base_model.load_state_dict(torch.load("outputs/models/lstm_baseline.pth", weights_only=False))

    gat_preds, base_preds = None, None
    if command in ("eval", "all"):
        gat_preds, base_preds = cmd_eval(cfg, data, gat_model, base_model, adj, ei, ew)

    if command in ("backtest", "all"):
        if gat_preds is None:
            gat_preds = predict(gat_model, data["test_X"], adj=adj, edge_weight=ew, is_gat=True, device=cfg.device)
            base_preds = predict(base_model, data["test_X"], is_gat=False, device=cfg.device)
        cmd_backtest(cfg, data, gat_preds, base_preds)

    print("\n" + "=" * 70)
    print("  ✅ DONE — All outputs saved to outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
