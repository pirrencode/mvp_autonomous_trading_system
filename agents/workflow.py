from __future__ import annotations

import json
from typing import Any, Dict, List

from prompts import ANALYSIS_PROMPT_TEMPLATE, SYSTEM_PROMPT


def _heuristic_fallback(user_inputs: Dict[str, Any], summaries: Dict[str, Any]) -> str:
    symbols = ", ".join(summaries.keys())
    style = user_inputs["strategy_style"]
    risk = user_inputs["risk_tolerance"]
    return f"""
### Market Context
- Watchlist: **{symbols}**
- Strategy style: **{style}**
- Risk tolerance: **{risk}**
- This is a deterministic fallback analysis because no LLM credentials were available.

### Signal Interpretation
- Use trend + RSI + MACD alignment as primary signal gate.
- Favor setups where momentum confirms trend and volatility regime matches time horizon.

### Candidate Trade Plans
1. **Trend Continuation**
   - Entry: pullback toward 20-day MA with confirmation candle.
   - Stop: below recent swing low or 1.2x ATR.
   - Take Profit: 2x risk.
   - Position Size: risk 0.5%-1.0% of portfolio per trade.
   - Confidence: 62/100.
2. **Mean Reversion**
   - Entry: RSI oversold (<30) within intact higher timeframe trend.
   - Stop: below signal candle low or 1.0x ATR.
   - Take Profit: prior midpoint / VWAP zone.
   - Position Size: smaller size than trend setup due to counter-trend nature.
   - Confidence: 55/100.

### QA / Risks
- Assumptions: liquidity remains stable and no major event shock.
- Invalidation: break of structure plus momentum divergence.
- Risks: macro news, earnings, spread widening, overnight gaps.
""".strip()


def _openai_analysis(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.output_text


def _anthropic_analysis(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1400,
        system=system_prompt,
        temperature=0.3,
        messages=[{"role": "user", "content": user_prompt}],
    )
    chunks = [c.text for c in msg.content if getattr(c, "type", "") == "text"]
    return "\n".join(chunks)


def generate_agentic_analysis(
    user_inputs: Dict[str, Any],
    ticker_summaries: Dict[str, Any],
    llm_provider: str,
    llm_model: str,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
) -> str:
    user_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        asset_class=user_inputs["asset_class"],
        time_horizon=user_inputs["time_horizon"],
        strategy_style=user_inputs["strategy_style"],
        risk_tolerance=user_inputs["risk_tolerance"],
        capital=user_inputs["capital"],
        thesis=user_inputs["thesis"] or "N/A",
        data_summary=json.dumps(ticker_summaries, indent=2),
    )

    try:
        if llm_provider == "anthropic" and anthropic_api_key:
            return _anthropic_analysis(anthropic_api_key, llm_model, SYSTEM_PROMPT, user_prompt)
        if llm_provider == "openai" and openai_api_key:
            return _openai_analysis(openai_api_key, llm_model, SYSTEM_PROMPT, user_prompt)
        if openai_api_key:
            return _openai_analysis(openai_api_key, llm_model, SYSTEM_PROMPT, user_prompt)
        if anthropic_api_key:
            return _anthropic_analysis(anthropic_api_key, llm_model, SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        return f"LLM call failed ({exc}).\n\n" + _heuristic_fallback(user_inputs, ticker_summaries)

    return _heuristic_fallback(user_inputs, ticker_summaries)


def build_ticker_summary(signal_map: Dict[str, Dict[str, str]], risk_map: Dict[str, Any], latest_map: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for symbol in signal_map:
        out[symbol] = {
            "latest": latest_map[symbol],
            "signals": signal_map[symbol],
            "risk": risk_map[symbol],
        }
    return out
