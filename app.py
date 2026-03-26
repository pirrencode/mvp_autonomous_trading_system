from __future__ import annotations

from typing import Dict

import streamlit as st

from agents.workflow import build_ticker_summary, generate_agentic_analysis
from analytics.indicators import add_indicators, signal_snapshot, summarize_risk
from charts.plotly_charts import (
    correlation_heatmap,
    indicator_charts,
    normalized_performance,
    price_chart,
    risk_charts,
    volume_chart,
)
from config import load_config
from data.providers import DataManager, resolve_provider


st.set_page_config(page_title="Autonomous Trading System", layout="wide")
st.title("🤖 Autonomous Trading System")
st.caption("Agentic AI MVP for market analysis and trade planning")

cfg = load_config()
provider, provider_name = resolve_provider(cfg.data_provider, cfg.alpha_vantage_api_key, cfg.finnhub_api_key)
manager = DataManager(provider)

with st.sidebar:
    st.header("Inputs")
    raw_symbols = st.text_input("Ticker symbol(s) or watchlist", value="AAPL,MSFT,BTC-USD")
    asset_class = st.selectbox("Asset class", ["stocks", "ETFs", "crypto", "forex"])
    time_horizon = st.selectbox("Time horizon", ["intraday", "swing", "long-term"])
    strategy_style = st.selectbox("Strategy style", ["momentum", "mean reversion", "breakout", "trend following"])
    risk_tolerance = st.selectbox("Risk tolerance", ["low", "medium", "high"])
    capital = st.number_input("Capital allocation / portfolio size", min_value=1000, value=100000, step=1000)
    thesis = st.text_area("Optional custom market thesis / notes", value="")
    date_window = st.select_slider("Date range", options=[90, 180, 365, 730], value=365, format_func=lambda x: f"{x} days")
    interval = st.selectbox("Bar interval", ["1d", "1h"], index=0)
    chart_style = st.radio("Price chart style", ["Candlestick", "Line"], horizontal=True)
    hide_data_warnings = st.checkbox("Hide API/rate-limit warnings (demo)", value=True)
    run = st.button("Run Agentic Workflow", type="primary")

st.info(f"Data source: **{provider_name}**. App will automatically use demo mode if API/data retrieval fails.")

if run:
    symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
    if not symbols:
        st.error("Please enter at least one ticker symbol.")
        st.stop()

    quotes, history_map, errors = manager.fetch_watchlist(symbols=symbols, days=date_window, interval=interval, asset_class=asset_class)

    if errors and not hide_data_warnings:
        st.warning("Some symbols failed to load: " + " | ".join(errors))
    elif errors and hide_data_warnings:
        st.caption("Some symbols could not be loaded. Warnings are hidden in demo mode.")

    if not history_map:
        st.error("Unable to load market data for requested symbols.")
        st.stop()

    processed: Dict[str, any] = {}
    risk_map: Dict[str, dict] = {}
    signal_map: Dict[str, dict] = {}
    latest_map: Dict[str, dict] = {}

    for sym, df in history_map.items():
        pdf = add_indicators(df)
        processed[sym] = pdf
        risk = summarize_risk(pdf)
        signals = signal_snapshot(pdf)
        risk_map[sym] = {
            "annualized_vol": round(risk.annualized_vol, 4),
            "max_drawdown": round(risk.max_drawdown, 4),
            "atr": round(risk.atr, 4),
            "regime": risk.regime,
        }
        signal_map[sym] = signals
        latest_map[sym] = {
            "close": float(pdf["close"].iloc[-1]),
            "volume": float(pdf["volume"].iloc[-1]),
            "rsi": float(pdf["rsi"].iloc[-1]) if not pdf["rsi"].isna().all() else 50.0,
            "macd": float(pdf["macd"].iloc[-1]) if not pdf["macd"].isna().all() else 0.0,
        }

    primary = list(processed.keys())[0]

    st.subheader("Overview")
    cols = st.columns(4)
    q = quotes.get(primary)
    if q:
        cols[0].metric(f"{primary} Price", f"{q.price:,.2f}")
        cols[1].metric("Daily Change", f"{q.change_pct:.2f}%")
        cols[2].metric("Volume", f"{q.volume:,.0f}" if q.volume else "N/A")
        cols[3].metric("Market Cap", f"{q.market_cap:,.0f}" if q.market_cap else "N/A")

    st.subheader("Price Action")
    p1, p2, p3 = st.columns(3)
    vol_fig, dd_fig = risk_charts(processed[primary], primary)
    p1.plotly_chart(
        price_chart(processed[primary], primary, use_candles=(chart_style == "Candlestick")),
        use_container_width=True,
    )
    p2.plotly_chart(volume_chart(processed[primary], primary), use_container_width=True)
    p3.plotly_chart(vol_fig, use_container_width=True)

    st.subheader("Risk and Technical Signals")
    v1, v2, v3 = st.columns(3)
    rsi_fig, macd_fig = indicator_charts(processed[primary], primary)
    v1.plotly_chart(dd_fig, use_container_width=True)
    v2.plotly_chart(rsi_fig, use_container_width=True)
    v3.plotly_chart(macd_fig, use_container_width=True)
    st.json(risk_map[primary])
    st.write(signal_map[primary])

    if len(processed) > 1:
        st.subheader("Comparative / Multi-Asset View")
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(normalized_performance(processed), use_container_width=True)
        c2.plotly_chart(correlation_heatmap(processed), use_container_width=True)
        c3.plotly_chart(price_chart(processed[primary], primary, use_candles=False), use_container_width=True)

    st.subheader("AI Strategy Workspace")
    ticker_summary = build_ticker_summary(signal_map, risk_map, latest_map)
    user_inputs = {
        "asset_class": asset_class,
        "time_horizon": time_horizon,
        "strategy_style": strategy_style,
        "risk_tolerance": risk_tolerance,
        "capital": capital,
        "thesis": thesis,
    }
    analysis_text = generate_agentic_analysis(
        user_inputs=user_inputs,
        ticker_summaries=ticker_summary,
        llm_provider=cfg.llm_provider,
        llm_model=cfg.llm_model,
        openai_api_key=cfg.openai_api_key,
        anthropic_api_key=cfg.anthropic_api_key,
    )

    st.markdown(analysis_text)
    st.code(analysis_text, language="markdown")
    st.download_button("Copy/Download AI Output", data=analysis_text, file_name="trade_plan.md", mime="text/markdown")
else:
    st.write("Configure your inputs and click **Run Agentic Workflow**.")
