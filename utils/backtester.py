"""
Backtesting Engine — updated for 3-class direction.

Uses argmax of softmax probs:
  class 2 (up)   + high confidence → go long
  class 0 (down) + high confidence → go short
  class 1 (flat) or low confidence → stay flat

4 strategies: GAT-LSTM, LSTM baseline, MA Crossover, Buy & Hold
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    exit_reason: str


class Backtester:

    def __init__(self, cfg_bt):
        self.initial_capital = cfg_bt.initial_capital
        self.leverage = cfg_bt.leverage
        self.spread = cfg_bt.spread_pips * cfg_bt.pip_value
        self.commission_pct = cfg_bt.commission_pct
        self.max_risk = cfg_bt.max_risk_per_trade
        self.confidence_threshold = cfg_bt.confidence_threshold
        self.atr_sl_mult = cfg_bt.atr_sl_multiplier
        self.atr_tp_mult = cfg_bt.atr_tp_multiplier
        self.max_dd_pct = cfg_bt.max_drawdown_pct

    def run_gat_strategy(self, df, dir_probs, ret_preds, seq_offset=30):
        """dir_probs is (N, 3) softmax for [down, flat, up]."""
        trade_df = df.iloc[seq_offset:].copy()
        n = min(len(trade_df), len(dir_probs))
        trade_df = trade_df.iloc[:n]
        return self._simulate(trade_df, dir_probs[:n], ret_preds[:n], "GAT-LSTM")

    def run_lstm_strategy(self, df, dir_probs, ret_preds, seq_offset=30):
        trade_df = df.iloc[seq_offset:].copy()
        n = min(len(trade_df), len(dir_probs))
        trade_df = trade_df.iloc[:n]
        return self._simulate(trade_df, dir_probs[:n], ret_preds[:n], "LSTM Baseline")

    def run_ma_crossover(self, df, fast=50, slow=200):
        trade_df = df.copy()
        ema_fast = trade_df["Close"].ewm(span=fast).mean()
        ema_slow = trade_df["Close"].ewm(span=slow).mean()
        signal = np.where(ema_fast > ema_slow, 1.0, -1.0)

        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        position = 0
        entry_price = 0.0

        for i in range(1, len(trade_df)):
            price = trade_df["Close"].iloc[i]
            prev_price = trade_df["Close"].iloc[i - 1]
            if position != 0:
                daily_ret = (price - prev_price) / prev_price * position
                capital *= (1 + daily_ret)
            if signal[i] != position:
                if position != 0:
                    pnl = (price - entry_price) / entry_price * position * self.initial_capital
                    trades.append(Trade(
                        str(trade_df.index[max(0, i-5)].date()), str(trade_df.index[i].date()),
                        "long" if position > 0 else "short", entry_price, price, 1.0, pnl, "signal_change"))
                position = signal[i]
                entry_price = price
            equity_curve.append(capital)
        return self._build_results(equity_curve, trades, "MA Crossover", trade_df)

    def run_buy_and_hold(self, df):
        prices = df["Close"].values
        daily_rets = np.diff(prices) / prices[:-1]
        capital = self.initial_capital
        equity_curve = [capital]
        for r in daily_rets:
            capital *= (1 + r * self.leverage)
            equity_curve.append(capital)
        trades = [Trade(
            str(df.index[0].date()), str(df.index[-1].date()),
            "long", prices[0], prices[-1], self.leverage,
            equity_curve[-1] - self.initial_capital, "end_of_period")]
        return self._build_results(equity_curve, trades, "Buy & Hold", df)

    # ──────────────────────────────────────────────────────────────────
    # Internal simulation (3-class direction)
    # ──────────────────────────────────────────────────────────────────

    def _simulate(self, df, dir_probs, ret_preds, strategy_name):
        """
        dir_probs: (N, 3) — [p_down, p_flat, p_up]
        Trade when max class prob > confidence_threshold and class != flat.
        """
        capital = self.initial_capital
        peak = capital
        equity_curve = [capital]
        trades: List[Trade] = []

        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        entry_idx = 0
        pos_size = 0.0
        circuit_breaker = False

        for i in range(1, len(df)):
            price = df["Close"].iloc[i]
            atr = df["ATR_14"].iloc[i] if "ATR_14" in df.columns else price * 0.005

            # Update equity
            if position != 0:
                daily_pnl = (price - df["Close"].iloc[i - 1]) * position * pos_size
                capital += daily_pnl

            # SL / TP check
            if position != 0:
                hit_sl = (position == 1 and price <= stop_loss) or (position == -1 and price >= stop_loss)
                hit_tp = (position == 1 and price >= take_profit) or (position == -1 and price <= take_profit)
                if hit_sl or hit_tp:
                    exit_price = stop_loss if hit_sl else take_profit
                    gross_pnl = (exit_price - entry_price) * position * pos_size
                    cost = self.spread * pos_size + abs(gross_pnl) * self.commission_pct
                    net_pnl = gross_pnl - cost
                    trades.append(Trade(
                        str(df.index[entry_idx].date()), str(df.index[i].date()),
                        "long" if position == 1 else "short",
                        entry_price, exit_price, pos_size, net_pnl,
                        "stop_loss" if hit_sl else "take_profit"))
                    capital += net_pnl - daily_pnl
                    position = 0

            # Circuit breaker
            peak = max(peak, capital)
            if (peak - capital) / peak > self.max_dd_pct:
                circuit_breaker = True

            # New signal
            if position == 0 and not circuit_breaker and i < len(dir_probs):
                probs = dir_probs[i]  # [p_down, p_flat, p_up]
                pred_class = int(np.argmax(probs))
                pred_conf = probs[pred_class]

                if pred_conf > self.confidence_threshold and pred_class != 1:
                    direction = 1 if pred_class == 2 else -1  # up→long, down→short
                    entry_price = price + (self.spread / 2 * direction)

                    risk_amount = capital * self.max_risk
                    sl_distance = atr * self.atr_sl_mult
                    if sl_distance > 0:
                        pos_size = min(risk_amount / sl_distance, capital * self.leverage / price)
                    else:
                        pos_size = capital * 0.01 / price

                    stop_loss = entry_price - direction * sl_distance
                    take_profit = entry_price + direction * atr * self.atr_tp_mult
                    position = direction
                    entry_idx = i

            equity_curve.append(capital)

        # Close open position
        if position != 0:
            exit_price = df["Close"].iloc[-1]
            gross_pnl = (exit_price - entry_price) * position * pos_size
            trades.append(Trade(
                str(df.index[entry_idx].date()), str(df.index[-1].date()),
                "long" if position == 1 else "short",
                entry_price, exit_price, pos_size, gross_pnl, "end_of_period"))

        return self._build_results(equity_curve, trades, strategy_name, df)

    def _build_results(self, equity_curve, trades, name, df):
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]

        total_ret = (equity[-1] - equity[0]) / equity[0]
        n_days = len(returns)
        ann = 252

        ann_ret = (1 + total_ret) ** (ann / max(n_days, 1)) - 1
        ann_vol = returns.std() * np.sqrt(ann) if len(returns) > 1 else 0
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        downside = returns[returns < 0]
        sortino = ann_ret / (downside.std() * np.sqrt(ann)) if len(downside) > 1 else 0

        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        max_dd = drawdowns.max()
        calmar = ann_ret / max_dd if max_dd > 0 else 0

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "name": name, "equity_curve": equity, "trades": trades,
            "total_return": total_ret, "annualized_return": ann_ret,
            "annualized_volatility": ann_vol, "sharpe_ratio": sharpe,
            "sortino_ratio": sortino, "max_drawdown": max_dd, "calmar_ratio": calmar,
            "num_trades": len(trades),
            "win_rate": len(wins) / max(len(trades), 1),
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else float("inf"),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "total_pnl": sum(pnls),
        }


def format_backtest_table(results_list):
    rows = []
    for r in results_list:
        rows.append({
            "Strategy": r["name"],
            "Total Return": f"{r['total_return']:.2%}",
            "Ann. Return": f"{r['annualized_return']:.2%}",
            "Sharpe": f"{r['sharpe_ratio']:.2f}",
            "Sortino": f"{r['sortino_ratio']:.2f}",
            "Max DD": f"{r['max_drawdown']:.2%}",
            "Calmar": f"{r['calmar_ratio']:.2f}",
            "Trades": r["num_trades"],
            "Win Rate": f"{r['win_rate']:.1%}",
            "Profit Factor": f"{r['profit_factor']:.2f}",
            "Total P&L": f"${r['total_pnl']:,.0f}",
        })
    return pd.DataFrame(rows)
