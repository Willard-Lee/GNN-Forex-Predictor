"""
Configuration for GAT-LSTM EUR/USD Forex Forecasting System.

Updated pipeline — 25 indicators (24 nodes + AD_Line),
multi-correlation graph edges (Pearson + DCC + Granger),
upgraded GAT (8→4 heads, 64→128 dim), 3-class direction,
cosine annealing, LayerNorm.
"""

from dataclasses import dataclass, field
from typing import List

from dataclasses import dataclass, field
from typing import List
import pandas as pd


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    
    csv_path: str = "data/EURUSD_daily.csv"
    separator: str = "\t"
    
    # Standardized OHLCV names
    date_col: str = "Date"
    ohlcv_cols: List[str] = field(default_factory=lambda: [
        "Open", "High", "Low", "Close", "Volume"
    ])

    # ── 25 technical indicator nodes ──
    # Momentum (10) | Volatility (6) | Trend (6) | Volume (3)
    feature_nodes: List[str] = field(default_factory=lambda: [
        # Momentum
        "RSI_14", "MACD", "MACD_Signal", "Stochastic_K", "Stochastic_D",
        "Williams_R", "CCI", "ROC", "MFI", "CMO",
        
        # Volatility
        "ATR_14", "ATR_20", "BB_Width", "ADX", "NATR", "StdDev_20",
        
        # Trend
        "EMA_20", "EMA_50", "SMA_20", "SMA_50",
        "Ichimoku_Tenkan", "Ichimoku_Kijun",
        
        # Volume
        "OBV", "CMF", "AD_Line",
    ])

    # Feature augmentation per node
    node_aug_dims: int = 4

    # Chronological walk-forward split
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # Sequence length
    sequence_length: int = 30

    # Target engineering
    direction_threshold: float = 0.002
    forecast_horizon: int = 1


def load_forex_data(config: DataConfig) -> pd.DataFrame:
    """Load EURUSD daily CSV and normalize column names."""
    
    df = pd.read_csv(config.csv_path, sep=config.separator)

    # Normalize raw MetaTrader column names
    rename_map = {
        "<DATE>": "Date",
        "<OPEN>": "Open",
        "<HIGH>": "High",
        "<LOW>": "Low",
        "<CLOSE>": "Close",
        "<TICKVOL>": "Volume"
    }

    df = df.rename(columns=rename_map)

    # Keep only normalized OHLCV
    required_cols = [config.date_col] + config.ohlcv_cols
    df = df[required_cols].copy()

    # Datetime conversion
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df
@dataclass
class GraphConfig:
    """Dynamic multi-edge graph construction."""
    # Primary: 30-day Pearson (|ρ| > 0.45)
    pearson_window: int = 30
    pearson_threshold: float = 0.45

    # Secondary: 7-day DCC-GARCH proxy (vol-adjusted short correlation)
    dcc_window: int = 7

    # Tertiary: Granger causality (p < 0.05, directional)
    granger_max_lag: int = 5
    granger_p_threshold: float = 0.05

    # Edge weight mix: 40/40/20
    weight_pearson: float = 0.40
    weight_dcc: float = 0.40
    weight_granger: float = 0.20

    # Sparsification
    top_k: int = 6
    min_edge_weight: float = 0.05


@dataclass
class ModelConfig:
    """GAT-LSTM architecture — upgraded per pipeline spec."""
    # GAT spatial block
    gat_hidden: int = 64
    gat_output: int = 128
    gat_heads_layer1: int = 8   # concat
    gat_heads_layer2: int = 4   # average
    gat_dropout: float = 0.1
    edge_dropout: float = 0.15

    # LSTM temporal block
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.1

    # Shared embedding before heads
    shared_dim: int = 128

    # Regularisation
    dropout: float = 0.1
    use_layer_norm: bool = True

    # 3-class direction (up / flat / down)
    num_direction_classes: int = 3


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15

    # Multi-task loss weights (pipeline spec)
    loss_weight_return: float = 0.4      # MSE
    loss_weight_volatility: float = 0.4  # MAE
    loss_weight_direction: float = 0.2   # CrossEntropy (3-class)

    # Scheduler
    scheduler: str = "cosine"

    # Reproducibility
    seed: int = 42


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100_000.0
    leverage: float = 10.0
    spread_pips: float = 1.5
    pip_value: float = 0.0001
    commission_pct: float = 0.0
    max_risk_per_trade: float = 0.02
    confidence_threshold: float = 0.55
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0
    max_drawdown_pct: float = 0.20


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    device: str = "cpu"
