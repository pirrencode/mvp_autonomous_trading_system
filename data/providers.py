from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


@dataclass
class Quote:
    symbol: str
    price: float
    change_pct: float
    volume: Optional[float]
    market_cap: Optional[float]


class MarketDataProvider(ABC):
    @abstractmethod
    def get_quote(self, symbol: str, asset_class: str = "stocks") -> Quote:
        raise NotImplementedError

    @abstractmethod
    def get_history(self, symbol: str, days: int = 365, interval: str = "1d", asset_class: str = "stocks") -> pd.DataFrame:
        raise NotImplementedError


class AlphaVantageProvider(MarketDataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, params: Dict[str, str]) -> Dict:
        base = "https://www.alphavantage.co/query"
        response = requests.get(base, params={**params, "apikey": self.api_key}, timeout=20)
        response.raise_for_status()
        return response.json()

    @st.cache_data(ttl=180)
    def get_quote(self, symbol: str, asset_class: str = "stocks") -> Quote:
        data = self._request({"function": "GLOBAL_QUOTE", "symbol": symbol})
        q = data.get("Global Quote", {})
        price = float(q.get("05. price", 0.0))
        change_pct = float(q.get("10. change percent", "0").replace("%", "") or 0.0)
        volume = float(q.get("06. volume", 0.0) or 0.0)
        return Quote(symbol=symbol, price=price, change_pct=change_pct, volume=volume, market_cap=None)

    @st.cache_data(ttl=900)
    def get_history(self, symbol: str, days: int = 365, interval: str = "1d", asset_class: str = "stocks") -> pd.DataFrame:
        data = self._request({"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol, "outputsize": "full"})
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            raise ValueError(f"No Alpha Vantage history returned for {symbol}")

        rows = []
        for date_str, row in ts.items():
            rows.append(
                {
                    "date": pd.to_datetime(date_str),
                    "open": float(row["1. open"]),
                    "high": float(row["2. high"]),
                    "low": float(row["3. low"]),
                    "close": float(row["4. close"]),
                    "volume": float(row["6. volume"]),
                }
            )
        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        return df[df.index >= (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days))]


class YFinanceProvider(MarketDataProvider):
    @st.cache_data(ttl=120)
    def get_quote(self, symbol: str, asset_class: str = "stocks") -> Quote:
        t = yf.Ticker(symbol)
        info = t.fast_info
        hist = t.history(period="2d", interval="1d")
        if len(hist) < 2:
            change_pct = 0.0
        else:
            change_pct = ((hist["Close"].iloc[-1] / hist["Close"].iloc[-2]) - 1) * 100
        return Quote(
            symbol=symbol,
            price=float(info.get("lastPrice") or hist["Close"].iloc[-1]),
            change_pct=float(change_pct),
            volume=float(info.get("lastVolume") or 0),
            market_cap=float(info.get("marketCap") or 0) if info.get("marketCap") else None,
        )

    @st.cache_data(ttl=600)
    def get_history(self, symbol: str, days: int = 365, interval: str = "1d", asset_class: str = "stocks") -> pd.DataFrame:
        start = datetime.utcnow() - timedelta(days=days)
        df = yf.download(symbol, start=start, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No yfinance history for {symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["open", "high", "low", "close", "volume"]]


class DemoProvider(MarketDataProvider):
    @st.cache_data(ttl=3600)
    def get_history(self, symbol: str, days: int = 365, interval: str = "1d", asset_class: str = "stocks") -> pd.DataFrame:
        rng = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D")
        seed = abs(hash(symbol)) % (2**32)
        rs = np.random.RandomState(seed)
        drift = 0.0003
        vol = 0.02
        rets = rs.normal(drift, vol, len(rng))
        close = 100 * np.exp(np.cumsum(rets))
        open_ = np.r_[close[0], close[:-1]] * (1 + rs.normal(0, 0.002, len(rng)))
        high = np.maximum(open_, close) * (1 + np.abs(rs.normal(0, 0.01, len(rng))))
        low = np.minimum(open_, close) * (1 - np.abs(rs.normal(0, 0.01, len(rng))))
        volume = rs.randint(5e5, 5e6, len(rng))
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=rng)

    @st.cache_data(ttl=600)
    def get_quote(self, symbol: str, asset_class: str = "stocks") -> Quote:
        hist = self.get_history(symbol, days=5)
        price = float(hist["close"].iloc[-1])
        change_pct = float((hist["close"].iloc[-1] / hist["close"].iloc[-2] - 1) * 100)
        return Quote(symbol=symbol, price=price, change_pct=change_pct, volume=float(hist["volume"].iloc[-1]), market_cap=None)


class DataManager:
    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def fetch_watchlist(self, symbols: List[str], days: int, interval: str, asset_class: str) -> Tuple[Dict[str, Quote], Dict[str, pd.DataFrame], List[str]]:
        quotes: Dict[str, Quote] = {}
        history: Dict[str, pd.DataFrame] = {}
        errors: List[str] = []
        for s in symbols:
            try:
                quotes[s] = self.provider.get_quote(s, asset_class)
                history[s] = self.provider.get_history(s, days=days, interval=interval, asset_class=asset_class)
            except Exception as exc:
                errors.append(f"{s}: {exc}")
        return quotes, history, errors


def resolve_provider(provider_name: str, alpha_key: Optional[str], finnhub_key: Optional[str]) -> tuple[MarketDataProvider, str]:
    if provider_name == "alpha_vantage" and alpha_key:
        return AlphaVantageProvider(alpha_key), "Alpha Vantage"
    if provider_name == "yfinance":
        return YFinanceProvider(), "yfinance"
    if provider_name == "demo":
        return DemoProvider(), "Demo"

    if alpha_key:
        return AlphaVantageProvider(alpha_key), "Alpha Vantage"
    try:
        return YFinanceProvider(), "yfinance"
    except Exception:
        return DemoProvider(), "Demo"
