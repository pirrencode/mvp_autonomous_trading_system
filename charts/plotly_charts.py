from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def price_chart(df: pd.DataFrame, symbol: str, use_candles: bool = True) -> go.Figure:
    fig = go.Figure()
    if use_candles:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=symbol,
            )
        )
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name=symbol))

    for ma in ["ma20", "ma50", "ma200"]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode="lines", name=ma.upper()))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10), title=f"{symbol} Price Action")
    return fig


def volume_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = px.bar(df, x=df.index, y="volume", title=f"{symbol} Volume")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def risk_charts(df: pd.DataFrame, symbol: str) -> tuple[go.Figure, go.Figure]:
    vol_fig = px.line(df, x=df.index, y="rolling_vol_30", title=f"{symbol} Rolling Volatility (30d)")
    dd_fig = px.area(df, x=df.index, y="drawdown", title=f"{symbol} Drawdown")
    vol_fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    dd_fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return vol_fig, dd_fig


def indicator_charts(df: pd.DataFrame, symbol: str) -> tuple[go.Figure, go.Figure]:
    rsi_fig = px.line(df, x=df.index, y="rsi", title=f"{symbol} RSI")
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal"))
    macd_fig.update_layout(title=f"{symbol} MACD", height=260, margin=dict(l=10, r=10, t=40, b=10))
    rsi_fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return rsi_fig, macd_fig


def normalized_performance(history: Dict[str, pd.DataFrame]) -> go.Figure:
    norm_df = pd.DataFrame({s: df["close"] / df["close"].iloc[0] for s, df in history.items()})
    fig = px.line(norm_df, x=norm_df.index, y=norm_df.columns, title="Normalized Performance")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def correlation_heatmap(history: Dict[str, pd.DataFrame]) -> go.Figure:
    rets = pd.DataFrame({s: df["close"].pct_change() for s, df in history.items()}).dropna(how="all")
    corr = rets.corr().fillna(0)
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig
