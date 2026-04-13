"""
Data Pipeline for GAT-LSTM EUR/USD Forecasting.

25 technical indicators computed from OHLCV.
Pipeline order (no data leakage):
  Raw CSV → 25 Indicators → Targets (return + vol + 3-class dir)
  → Chronological Split → QuantileTransform (fit train) → Sequences
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from typing import Tuple, Dict
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 1. Load & validate
# ---------------------------------------------------------------------------

def load_data(csv_path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load OHLCV data from CSV and validate."""
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Ensure Volume exists (fill 0 if missing — forex often has no volume)
    if "Volume" not in df.columns:
        df["Volume"] = 0
        print("  ℹ️  No Volume column — filled with 0 (volume indicators will be flat)")

    print(f"✅ Loaded {len(df)} rows  |  {df.index.min().date()} → {df.index.max().date()}")
    return df


# ---------------------------------------------------------------------------
# 2. Compute 25 technical indicators
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 25 technical indicators from OHLCV.
    All computations are strictly causal (no future leakage).
    No TA-Lib dependency — pure pandas/numpy.
    """
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"].astype(float)
    hlc3 = (high + low + close) / 3

    # ═══════════════════════════════════════════════════════════════════
    # MOMENTUM (10)
    # ═══════════════════════════════════════════════════════════════════

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss_s = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss_s.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # Stochastic K/D (14, 3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    out["Stochastic_K"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    out["Stochastic_D"] = out["Stochastic_K"].rolling(3).mean()

    # Williams %R (14)
    out["Williams_R"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)

    # CCI (20)
    tp = hlc3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out["CCI"] = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)

    # ROC (12)
    out["ROC"] = close.pct_change(12) * 100

    # MFI (14) — Money Flow Index
    tp_val = hlc3
    raw_mf = tp_val * volume
    mf_pos = raw_mf.where(tp_val > tp_val.shift(1), 0)
    mf_neg = raw_mf.where(tp_val < tp_val.shift(1), 0)
    mfr = mf_pos.rolling(14).sum() / mf_neg.rolling(14).sum().replace(0, np.nan)
    out["MFI"] = 100 - (100 / (1 + mfr))

    # CMO (14) — Chande Momentum Oscillator
    mom = close.diff()
    su = mom.clip(lower=0).rolling(14).sum()
    sd = (-mom.clip(upper=0)).rolling(14).sum()
    out["CMO"] = 100 * (su - sd) / (su + sd).replace(0, np.nan)

    # ═══════════════════════════════════════════════════════════════════
    # VOLATILITY (6)
    # ═══════════════════════════════════════════════════════════════════

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # ATR(14) and ATR(20)
    out["ATR_14"] = tr.rolling(14).mean()
    out["ATR_20"] = tr.rolling(20).mean()

    # Bollinger Width (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    out["BB_Width"] = (bb_upper - bb_lower) / sma20.replace(0, np.nan)

    # ADX (14)
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr14_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["ADX"] = dx.ewm(span=14, adjust=False).mean()

    # NATR (14) — Normalized ATR
    out["NATR"] = (out["ATR_14"] / close) * 100

    # StdDev(20)
    out["StdDev_20"] = close.rolling(20).std()

    # ═══════════════════════════════════════════════════════════════════
    # TREND (6)
    # ═══════════════════════════════════════════════════════════════════

    out["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    out["SMA_20"] = close.rolling(20).mean()
    out["SMA_50"] = close.rolling(50).mean()

    # Ichimoku Tenkan-sen (9) and Kijun-sen (26)
    out["Ichimoku_Tenkan"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    out["Ichimoku_Kijun"] = (high.rolling(26).max() + low.rolling(26).min()) / 2

    # ═══════════════════════════════════════════════════════════════════
    # VOLUME (3)
    # ═══════════════════════════════════════════════════════════════════

    # OBV — On-Balance Volume
    obv_sign = np.sign(close.diff()).fillna(0)
    out["OBV"] = (volume * obv_sign).cumsum()

    # CMF — Chaikin Money Flow (20)
    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_volume = mf_multiplier * volume
    out["CMF"] = mf_volume.rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)

    # A/D Line — Accumulation/Distribution
    out["AD_Line"] = mf_volume.cumsum()

    # Drop NaN warm-up rows (longest warm-up is Ichimoku Kijun = 26 + SMA50)
    out = out.dropna()
    print(f"✅ 25 indicators computed  |  {len(out)} rows after warm-up drop")
    return out


# ---------------------------------------------------------------------------
# 3. Targets — return, volatility, 3-class direction
# ---------------------------------------------------------------------------

def create_targets(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.002) -> pd.DataFrame:
    """
    Create prediction targets:
      - target_return:    log(close_{t+h} / close_t)
      - target_volatility: 5-day rolling std dev of log returns
      - target_direction:  0=down, 1=flat, 2=up  (thresholds ±0.2%)
    """
    out = df.copy()
    future_close = out["Close"].shift(-horizon)
    log_ret = np.log(future_close / out["Close"])

    out["target_return"] = log_ret
    out["target_volatility"] = np.log(out["Close"] / out["Close"].shift(1)).rolling(5).std()

    # 3-class direction
    out["target_direction"] = 1  # flat
    out.loc[log_ret > threshold, "target_direction"] = 2   # up
    out.loc[log_ret < -threshold, "target_direction"] = 0  # down
    out["target_direction"] = out["target_direction"].astype(int)

    out = out.dropna(subset=["target_return", "target_volatility", "target_direction"])

    counts = out["target_direction"].value_counts().sort_index()
    labels = {0: "Down", 1: "Flat", 2: "Up"}
    dist_str = " | ".join(f"{labels[i]}: {counts.get(i, 0)} ({counts.get(i, 0)/len(out)*100:.1f}%)" for i in [0, 1, 2])
    print(f"✅ Targets created  |  {len(out)} rows  |  {dist_str}")
    return out


# ---------------------------------------------------------------------------
# 4. Chronological split
# ---------------------------------------------------------------------------

def split_data(df, train_ratio=0.70, val_ratio=0.15):
    """Strictly chronological split — no shuffling."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    assert train.index.max() < val.index.min(), "Train/val overlap!"
    assert val.index.max() < test.index.min(), "Val/test overlap!"

    print(f"✅ Split  |  Train: {len(train)} ({train.index.min().date()}→{train.index.max().date()})  "
          f"|  Val: {len(val)}  |  Test: {len(test)}")
    return train, val, test


# ---------------------------------------------------------------------------
# 5. Quantile Transform (fit on train only)
# ---------------------------------------------------------------------------

def scale_features(train, val, test, feature_cols):
    """Quantile transform to [0, 1]. Fitted ONLY on training data."""
    scaler = QuantileTransformer(output_distribution="uniform", n_quantiles=min(1000, len(train)))

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    print(f"✅ Quantile-transformed to [0,1] (fit on train only)")
    return train, val, test, scaler


# ---------------------------------------------------------------------------
# 6. Sequence creation
# ---------------------------------------------------------------------------

def create_sequences(df, feature_cols, seq_len=30):
    """
    Create overlapping sequences.

    Returns:
        X: (N, seq_len, num_features)
        y: dict with 'direction' (int), 'return' (float), 'volatility' (float)
    """
    features = df[feature_cols].values
    dir_t = df["target_direction"].values
    ret_t = df["target_return"].values
    vol_t = df["target_volatility"].values

    X, y_dir, y_ret, y_vol = [], [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y_dir.append(dir_t[i])
        y_ret.append(ret_t[i])
        y_vol.append(vol_t[i])

    X = np.array(X, dtype=np.float32)
    targets = {
        "direction": np.array(y_dir, dtype=np.int64),
        "return": np.array(y_ret, dtype=np.float32),
        "volatility": np.array(y_vol, dtype=np.float32),
    }
    return X, targets


# ---------------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg) -> dict:
    """Execute the full data pipeline."""
    df = load_data(cfg.data.csv_path, cfg.data.date_col)
    df = compute_features(df)
    df = create_targets(df, cfg.data.forecast_horizon, cfg.data.direction_threshold)

    train_df, val_df, test_df = split_data(df, cfg.data.train_ratio, cfg.data.val_ratio)

    feat = cfg.data.feature_nodes
    train_df, val_df, test_df, scaler = scale_features(train_df, val_df, test_df, feat)

    train_X, train_y = create_sequences(train_df, feat, cfg.data.sequence_length)
    val_X, val_y = create_sequences(val_df, feat, cfg.data.sequence_length)
    test_X, test_y = create_sequences(test_df, feat, cfg.data.sequence_length)

    print(f"\n📊 Sequence shapes:")
    print(f"   Train: {train_X.shape}  |  Val: {val_X.shape}  |  Test: {test_X.shape}")

    return {
        "train_X": train_X, "train_y": train_y,
        "val_X": val_X, "val_y": val_y,
        "test_X": test_X, "test_y": test_y,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "scaler": scaler, "feature_cols": feat, "df_full": df,
    }
