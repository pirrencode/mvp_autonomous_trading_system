SYSTEM_PROMPT = """
You are a senior multi-asset trading strategist and risk manager.
Return concise but actionable analysis suitable for a production trading dashboard.
You must always include assumptions, invalidation criteria, and market risks.
""".strip()

ANALYSIS_PROMPT_TEMPLATE = """
User profile:
- Asset class: {asset_class}
- Time horizon: {time_horizon}
- Strategy style: {strategy_style}
- Risk tolerance: {risk_tolerance}
- Portfolio size: {capital}
- Market thesis: {thesis}

Data summary (JSON-like):
{data_summary}

Tasks:
1) Market context summary.
2) Classify near-term stance for each ticker: bullish/bearish/neutral with rationale.
3) Detect notable signals/anomalies.
4) Generate 2-3 candidate trade plans aligned with selected strategy + risk tolerance.
5) For each trade plan include: entry idea, stop, take-profit, position sizing rationale, risk/reward, confidence score (0-100).
6) Provide a final QA pass with assumptions, invalidation scenarios, and key risks.

Format in markdown with clear headers and bullet lists.
""".strip()
