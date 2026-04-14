"""
Streamlit Dashboard — GAT-LSTM EUR/USD Forecasting System 

Run: streamlit run app.py
Requires: outputs/ folder populated by `python main.py all`
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os

st.set_page_config(
    page_title="GAT-LSTM Forex Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.title("📈 GAT-LSTM Forex")
st.sidebar.markdown("**EUR/USD Forecasting System v2**")
st.sidebar.divider()

data_ok = os.path.exists("data/EURUSD_daily.csv")
models_ok = os.path.exists("outputs/models/gat_lstm.pth")
metrics_ok = os.path.exists("outputs/metrics.json")
bt_ok = os.path.exists("outputs/backtest_results.json")

st.sidebar.markdown("### System Status")
st.sidebar.markdown(f"{'✅' if data_ok else '❌'} Data loaded")
st.sidebar.markdown(f"{'✅' if models_ok else '❌'} Models trained")
st.sidebar.markdown(f"{'✅' if metrics_ok else '❌'} Evaluation complete")
st.sidebar.markdown(f"{'✅' if bt_ok else '❌'} Backtesting complete")

if not models_ok:
    st.sidebar.info("Run `python main.py all` first")

st.sidebar.divider()
st.sidebar.markdown("### Architecture")
st.sidebar.code(
    "Nodes:     25 indicators\n"
    "GAT L1:    8 heads, 64 dim (concat)\n"
    "GAT L2:    4 heads, 128 dim (avg)\n"
    "LSTM:      2 layers, 128 hidden\n"
    "Direction: 3-class (up/flat/down)\n"
    "Edges:     Pearson+DCC+Granger\n"
    "Loss:      0.4 MSE + 0.4 MAE + 0.2 CE",
    language=None,
)

# ── Tabs ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Predictions", "💰 Backtesting", "🏗️ Architecture"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("EUR/USD GAT-LSTM Forecasting — Overview")

    if data_ok:
        df = pd.read_csv("data/EURUSD_daily.csv", sep="\t")

        df = df.rename(columns={
            "<DATE>": "Date",
            "<OPEN>": "Open",
            "<HIGH>": "High",
            "<LOW>": "Low",
            "<CLOSE>": "Close",
            "<TICKVOL>": "Volume"
        })

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Data Period",
            f"{df.index.min().strftime('%Y-%m')} → {df.index.max().strftime('%Y-%m')}"
        )
        c2.metric("Total Days", f"{len(df):,}")
        c3.metric("Price Range", f"{df['Close'].min():.4f} – {df['Close'].max():.4f}")
        c4.metric("Latest Close", f"{df['Close'].iloc[-1]:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="EUR/USD Close",
            line=dict(color="#2563eb", width=1.5)
        ))

        n = len(df)
        te, ve = int(n * 0.70), int(n * 0.85)

        fig.add_vrect(
            x0=df.index[0], x1=df.index[te],
            fillcolor="blue", opacity=0.04,
            annotation_text="Train",
            annotation_position="top left"
        )

        fig.add_vrect(
            x0=df.index[te], x1=df.index[ve],
            fillcolor="orange", opacity=0.07,
            annotation_text="Val"
        )

        fig.add_vrect(
            x0=df.index[ve], x1=df.index[-1],
            fillcolor="green", opacity=0.07,
            annotation_text="Test"
        )

        fig.update_layout(
            title="EUR/USD Daily Close with Data Splits",
            xaxis_title="Date",
            yaxis_title="Price",
            height=450,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("25 Technical Indicator Nodes")

        ind_data = {
            "Category": ["Momentum"]*10 + ["Volatility"]*6 + ["Trend"]*6 + ["Volume"]*3,
            "Indicator": [
                "RSI(14)", "MACD", "MACD Signal", "Stochastic K", "Stochastic D",
                "Williams %R", "CCI", "ROC", "MFI", "CMO",
                "ATR(14)", "ATR(20)", "BB Width", "ADX", "NATR", "StdDev(20)",
                "EMA(20)", "EMA(50)", "SMA(20)", "SMA(50)", "Ichimoku Tenkan", "Ichimoku Kijun",
                "OBV", "CMF", "A/D Line",
            ],
            "Description": [
                "Relative Strength Index", "Moving Avg Convergence Divergence", "MACD signal line",
                "Stochastic oscillator %K", "Stochastic oscillator %D", "Williams percent range",
                "Commodity Channel Index", "Rate of Change", "Money Flow Index", "Chande Momentum Oscillator",
                "Average True Range 14d", "Average True Range 20d", "Bollinger Band width",
                "Average Directional Index", "Normalised ATR", "20-day standard deviation",
                "20-day EMA", "50-day EMA", "20-day SMA", "50-day SMA",
                "Ichimoku conversion line", "Ichimoku base line",
                "On-Balance Volume", "Chaikin Money Flow", "Accumulation/Distribution Line",
            ],
        }

        st.dataframe(pd.DataFrame(ind_data), use_container_width=True, hide_index=True)

    else:
        st.warning("Place `EURUSD_daily.csv` in `data/` and run `python main.py all`.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Model Evaluation — Test Set Results")

    if metrics_ok:
        with open("outputs/metrics.json") as f:
            metrics = json.load(f)

        gat_m = metrics["gat_lstm"]
        base_m = metrics["baseline_lstm"]
        sig = metrics.get("significance", {})

        st.subheader("GAT-LSTM vs Baseline LSTM")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### GAT-LSTM (25-node graph)")
            st.metric("3-Class Accuracy", f"{gat_m['accuracy']:.1%}")
            st.metric("F1 Macro", f"{gat_m['f1_macro']:.4f}")
            st.metric("Return RMSE", f"{gat_m['ret_rmse']:.6f}")
            st.metric("Volatility MAE", f"{gat_m['vol_mae']:.6f}")
        with c2:
            st.markdown("#### Baseline LSTM (no graph)")
            st.metric("3-Class Accuracy", f"{base_m['accuracy']:.1%}")
            st.metric("F1 Macro", f"{base_m['f1_macro']:.4f}")
            st.metric("Return RMSE", f"{base_m['ret_rmse']:.6f}")
            st.metric("Volatility MAE", f"{base_m['vol_mae']:.6f}")

        # Per-class F1 comparison
        fig = go.Figure(data=[
            go.Bar(name="GAT-LSTM", x=["Down", "Flat", "Up"],
                   y=[gat_m.get("f1_down", 0), gat_m.get("f1_flat", 0), gat_m.get("f1_up", 0)],
                   marker_color="#2563eb"),
            go.Bar(name="LSTM Baseline", x=["Down", "Flat", "Up"],
                   y=[base_m.get("f1_down", 0), base_m.get("f1_flat", 0), base_m.get("f1_up", 0)],
                   marker_color="#dc2626"),
        ])
        fig.update_layout(barmode="group", title="Per-Class F1 Score",
                          yaxis_title="F1", height=380, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Significance
        st.subheader("Statistical Significance (Paired t-test)")
        if sig:
            st.table(pd.DataFrame([
                {"Test": "Return RMSE",
                 "t-stat": f"{sig.get('return_t_stat', 0):.3f}",
                 "p-value": f"{sig.get('return_p_value', 1):.4f}",
                 "Significant": "✅" if sig.get("significant_return") else "❌"},
                {"Test": "Direction Accuracy",
                 "t-stat": f"{sig.get('direction_t_stat', 0):.3f}",
                 "p-value": f"{sig.get('direction_p_value', 1):.4f}",
                 "Significant": "✅" if sig.get("significant_direction") else "❌"},
            ]))

        # Confusion matrices
        if os.path.exists("outputs/confusion_matrices.png"):
            st.subheader("Confusion Matrices (Down / Flat / Up)")
            st.image("outputs/confusion_matrices.png")

        # Attention heatmap
        if os.path.exists("outputs/attention_heatmap.png"):
            st.subheader("GAT Attention Weights (25×25)")
            st.image("outputs/attention_heatmap.png",
                     caption="Which indicator relationships the GAT focuses on for prediction")

        # Training curves
        if os.path.exists("outputs/training_curves.png"):
            st.subheader("Training Curves")
            st.image("outputs/training_curves.png")
    else:
        st.info("Run `python main.py all` to generate evaluation results.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: BACKTESTING
# ══════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Backtesting Results — Test Period")

    if bt_ok:
        with open("outputs/backtest_results.json") as f:
            bt_results = json.load(f)

        rows = []
        for r in bt_results:
            rows.append({
                "Strategy": r["name"],
                "Total Return": f"{r['total_return']:.2%}",
                "Sharpe": f"{r['sharpe_ratio']:.2f}",
                "Sortino": f"{r['sortino_ratio']:.2f}",
                "Max DD": f"{r['max_drawdown']:.2%}",
                "Trades": r["num_trades"],
                "Win Rate": f"{r['win_rate']:.1%}",
                "P&L": f"${r['total_pnl']:,.0f}",
            })
        st.table(pd.DataFrame(rows))

        # Sharpe bar chart
        fig = go.Figure(data=[go.Bar(
            x=[r["name"] for r in bt_results],
            y=[r["sharpe_ratio"] for r in bt_results],
            marker_color=["#2563eb", "#dc2626", "#16a34a", "#9333ea"],
            text=[f"{r['sharpe_ratio']:.2f}" for r in bt_results],
            textposition="outside",
        )])
        fig.update_layout(title="Sharpe Ratio Comparison", yaxis_title="Sharpe", height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        if os.path.exists("outputs/backtest_results.png"):
            st.subheader("Equity Curves & Drawdown")
            st.image("outputs/backtest_results.png")

        with st.expander("ℹ️ Strategy Details"):
            st.markdown("""
            **GAT-LSTM**: Trades when 3-class directional confidence > 55% for up or down (ignores flat).
            Position sized by ATR. Stop-loss 2× ATR, take-profit 3× ATR. 10× leverage. 20% drawdown circuit breaker.

            **LSTM Baseline**: Same trading logic using baseline LSTM predictions (no graph).
            Ablation to isolate the GAT contribution.

            **MA Crossover**: 50/200 EMA crossover. No leverage. Long when fast > slow, short otherwise.

            **Buy & Hold**: Enter long at start, hold until end. 10× leverage.
            """)

        with st.expander("ℹ️ Risk Management"):
            st.markdown("""
            **ATR-Based Position Sizing**: Risk per trade = 2% of capital. Position size = risk / (ATR × 2).

            **3-Class Signal Filter**: Only trade on class 0 (down) or class 2 (up) with confidence > 55%.
            Class 1 (flat) predictions → no trade, reducing overtrading.

            **Circuit Breaker**: If drawdown exceeds 20% of peak equity, all trading halts.
            """)
    else:
        st.info("Run `python main.py all` to generate backtest results.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("System Architecture")

    st.subheader("Dual-Path GAT-LSTM Architecture (v2)")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        ### 🔷 Path A: GAT (Spatial)
        - **Input**: 25 nodes × 4 features
          (raw, z-score, 3d-slope, 5d-vol)
        - **GAT L1**: 64 dim, 8 heads → 512
        - **GAT L2**: 512 → 128 (4 heads, avg)
        - **Flatten**: 25 × 128 = 3,200
        - **LayerNorm** + edge dropout 15%

        *Learns which indicator relationships
        matter via attention mechanism*
        """)

    with c2:
        st.markdown("""
        ### 🔶 Path B: LSTM (Temporal)
        - **Input**: 30 timesteps × 25 features
        - **LSTM**: 2 layers, 128 hidden
        - **Output**: 128 (last hidden + LayerNorm)
        - **Dropout**: 10%

        *Captures sequential patterns
        and temporal dependencies*
        """)

    with c3:
        st.markdown("""
        ### 🟢 Fusion + 3 Heads
        - **Concat**: 3,200 + 128 = 3,328
        - **Shared**: 3,328 → 128 (LayerNorm, ReLU)
        - **Direction**: 128 → 3 (CrossEntropy)
        - **Return**: 128 → 1 (MSE)
        - **Volatility**: 128 → 1 (MAE)

        *Multi-task loss: 0.4 MSE + 0.4 MAE + 0.2 CE*
        """)

    st.divider()

    st.subheader("Multi-Edge Graph Construction")
    st.markdown("""
    | Edge Type | Window | Threshold | Weight | Purpose |
    |-----------|--------|-----------|--------|---------|
    | Pearson correlation | 30 days | \|ρ\| > 0.45 | 40% | Linear statistical dependency |
    | DCC-GARCH proxy | 7 days | — | 40% | Vol-adjusted short-term correlation |
    | Granger causality | 5 lags | p < 0.05 | 20% | Directional causal influence |

    Edges are sparsified via top-k (k=6) per node. Graph is rebuilt from **training data only**.
    """)

    st.divider()

    st.subheader("Pipeline Flow")
    st.code("""
    Raw OHLCV CSV
        ↓
    25 Technical Indicators (Momentum/Volatility/Trend/Volume)
        ↓
    Target Engineering (return + volatility + 3-class direction)
        ↓
    Chronological Split (70% train / 15% val / 15% test)
        ↓
    Quantile Transform to [0,1] (fit on TRAIN ONLY)
        ↓
    Multi-Edge Graph (Pearson + DCC + Granger, TRAIN ONLY)
        ↓
    30-day Sliding Window Sequences
        ↓
    Train GAT-LSTM + Baseline LSTM (AdamW + cosine annealing)
        ↓
    Evaluate (3-class F1, confusion matrix, return/vol metrics, t-test)
        ↓
    Backtest (4 strategies with realistic costs, ATR stops, circuit breaker)
    """, language=None)

    st.subheader("File Structure")
    st.code("""
    gat-forex/
    ├── main.py                  # CLI: train | eval | backtest | all
    ├── app.py                   # Streamlit dashboard (this file)
    ├── requirements.txt
    ├── configs/
    │   └── config.py            # All hyperparameters
    ├── data/
    │   └── EURUSD_daily.csv     # Your data
    ├── models/
    │   └── gat_lstm.py          # GAT-LSTM (937K params)
    ├── baselines/
    │   └── lstm_baseline.py     # LSTM ablation (229K params)
    ├── utils/
    │   ├── data_pipeline.py     # 25 indicators → targets → split → scale
    │   ├── graph_builder.py     # Pearson + DCC + Granger multi-edge graph
    │   ├── trainer.py           # AdamW + cosine annealing + 3-class CE
    │   ├── evaluator.py         # F1 macro/per-class, confusion matrix, t-test
    │   └── backtester.py        # 4 strategies, ATR stops, circuit breaker
    └── outputs/                 # Generated by main.py
        ├── models/              # .pth weights + graph.pth
        ├── metrics.json
        ├── backtest_results.json
        ├── training_curves.png
        ├── confusion_matrices.png
        ├── backtest_results.png
        └── attention_heatmap.png
    """, language=None)

    st.subheader("Key Design Decisions")
    st.markdown("""
    1. **25 Feature-as-Node Graph** — Each indicator is a node; edges represent multi-type correlations.
       Novel contribution: Pearson + DCC-GARCH + Granger causality weighted composite.

    2. **3-Class Direction** — Up/flat/down with ±0.2% thresholds reduces noise from small moves.
       The flat class acts as a natural trade filter — the model learns when NOT to trade.

    3. **No Data Leakage** — Graph, scaler, and features are strictly from training data.

    4. **Multi-Task Learning** — Return MSE + volatility MAE + direction CE (0.4/0.4/0.2 weights)
       provides richer gradient signal than any single task.

    5. **Manual GAT** — No PyTorch Geometric dependency. The 25-node graph is small enough
       for dense attention. Edge dropout (15%) regularises the graph structure.

    6. **937K Parameters** — Upgraded from 38K in v1, but still modest enough to avoid
       overfitting on ~1,800 training sequences (10 years daily data).
    """)
