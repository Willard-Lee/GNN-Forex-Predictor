# GAT-LSTM EUR/USD Forex Forecasting System 

A hybrid **Graph Attention Network + LSTM** model for EUR/USD daily forecasting using 25 technical indicators modelled as a dynamic multi-edge graph.

## Quick Start

```bash
pip install -r requirements.txt

# Place your data
cp /path/to/EURUSD_daily.csv data/

# Run full pipeline
python main.py all

# Launch dashboard
streamlit run app.py
```

## Commands


| Command                   | Description                                       |
| ------------------------- | ------------------------------------------------- |
| `python main.py train`    | Train GAT-LSTM + baseline LSTM                    |
| `python main.py eval`     | Evaluate on test set (metrics + confusion matrix) |
| `python main.py backtest` | Run 4 backtesting strategies                      |
| `python main.py all`      | Full pipeline                                     |


## Architecture

```
Path A: GAT (spatial)              Path B: LSTM (temporal)
  25 indicator nodes                 30-day sequence
  4 features per node                2-layer, 128 hidden
  (raw, z-score, 3d-slope, 5d-vol)  Last hidden state
  GAT L1: 8 heads concat (→512)
  GAT L2: 4 heads avg (→128)
  Flatten: 25×128 = 3,200
           ↓                              ↓
           └──── Concatenate (3,328) ─────┘
                        ↓
               Shared FC → 128 (LayerNorm)
                        ↓
           ┌────────────┼────────────┐
      Direction     Return       Volatility
      3-class CE    MSE          MAE
      (×0.2)        (×0.4)       (×0.4)
```

## Multi-Edge Graph


| Type              | Window | Threshold  | Weight | Purpose                        |
| ----------------- | ------ | ---------- | ------ | ------------------------------ |
| Pearson           | 30d    | |ρ| > 0.45 | 40%    | Linear correlation             |
| DCC-GARCH proxy   | 7d     | —          | 40%    | Vol-adjusted short correlation |
| Granger causality | 5 lags | p < 0.05   | 20%    | Directional causation          |


## Data Requirements

CSV with columns: `Date, Open, High, Low, Close, Volume`

All 25 indicators are computed internally — no TA-Lib dependency.

## Safeguards (No Data Leakage)

- Chronological split only (70/15/15), no shuffling
- QuantileTransformer fitted on training data only
- Graph built from training data only
- All indicator computations strictly causal

