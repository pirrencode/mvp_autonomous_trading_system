from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class RiskSummary:
    annualized_vol: float
    max_drawdown: float
    atr: float
    regime: str


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma20"] = out["close"].rolling(20).mean()
    out["ma50"] = out["close"].rolling(50).mean()
    out["ma200"] = out["close"].rolling(200).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    out["rsi"] = 100 - (100 / (1 + rs))

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    tr_components = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs(),
        ],
        axis=1,
    )
    out["tr"] = tr_components.max(axis=1)
    out["atr14"] = out["tr"].rolling(14).mean()

    out["returns"] = out["close"].pct_change()
    out["rolling_vol_30"] = out["returns"].rolling(30).std() * np.sqrt(252)

    running_max = out["close"].cummax()
    out["drawdown"] = (out["close"] / running_max) - 1

    return out


def summarize_risk(df: pd.DataFrame) -> RiskSummary:
    annualized_vol = float(df["returns"].dropna().std() * np.sqrt(252))
    max_drawdown = float(df["drawdown"].min())
    atr = float(df["atr14"].iloc[-1]) if pd.notna(df["atr14"].iloc[-1]) else 0.0

    if annualized_vol > 0.45:
        regime = "High volatility"
    elif annualized_vol > 0.25:
        regime = "Moderate volatility"
    else:
        regime = "Low volatility"

    return RiskSummary(annualized_vol=annualized_vol, max_drawdown=max_drawdown, atr=atr, regime=regime)


def signal_snapshot(df: pd.DataFrame) -> Dict[str, str]:
    latest = df.iloc[-1]
    trend = "Bullish" if latest["close"] > latest.get("ma50", latest["close"]) else "Bearish"
    rsi_val = float(latest.get("rsi", 50))
    if rsi_val >= 70:
        rsi_sig = "Overbought"
    elif rsi_val <= 30:
        rsi_sig = "Oversold"
    else:
        rsi_sig = "Neutral"

    macd_sig = "Bullish" if latest.get("macd", 0) > latest.get("macd_signal", 0) else "Bearish"
    return {
        "trend": trend,
        "rsi": rsi_sig,
        "macd": macd_sig,
        "momentum": "Positive" if latest.get("returns", 0) > 0 else "Negative",
    }
