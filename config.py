import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st


@dataclass
class AppConfig:
    data_provider: str = "auto"
    alpha_vantage_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"


def _from_streamlit_secrets(key: str) -> Optional[str]:
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def load_config() -> AppConfig:
    return AppConfig(
        data_provider=os.getenv("DATA_PROVIDER", "auto"),
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY") or _from_streamlit_secrets("ALPHA_VANTAGE_API_KEY"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY") or _from_streamlit_secrets("FINNHUB_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY") or _from_streamlit_secrets("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or _from_streamlit_secrets("ANTHROPIC_API_KEY"),
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    )
