# Autonomous Trading System (Streamlit MVP)

Production-style Streamlit MVP for agentic trading analysis.

## Features
- Dashboard with market overview, price action, risk, technicals, and multi-asset comparison.
- Modular data layer with swappable providers:
  - Alpha Vantage (real API)
  - yfinance (fallback)
  - deterministic demo/mock data mode
- Agentic AI workflow:
  1. fetch + normalize data
  2. compute indicators + risk metrics
  3. render charts and signal panels
  4. run LLM strategy analysis
  5. produce candidate trade plans + QA risk pass
- Editable prompt templates (`prompts.py`).
- OpenAI and Anthropic integrations (with graceful fallback to deterministic heuristic output).

## Project Structure
```text
app.py
config.py
data/providers.py
analytics/indicators.py
charts/plotly_charts.py
agents/workflow.py
prompts.py
requirements.txt
```

## Local Run
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables (optional, but recommended):
   ```bash
   export ALPHA_VANTAGE_API_KEY="..."
   export OPENAI_API_KEY="..."           # optional for LLM analysis
   export ANTHROPIC_API_KEY="..."        # optional alternative
   export DATA_PROVIDER="auto"            # auto | alpha_vantage | yfinance | demo
   export LLM_PROVIDER="openai"           # openai | anthropic
   export LLM_MODEL="gpt-4o-mini"
   ```
4. Run app:
   ```bash
   streamlit run app.py
   ```

## Streamlit Secrets
For Streamlit Community Cloud, store credentials in `.streamlit/secrets.toml`:
```toml
ALPHA_VANTAGE_API_KEY="..."
OPENAI_API_KEY="..."
ANTHROPIC_API_KEY="..."
```

## Deployment (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Create new app in Streamlit Community Cloud.
3. Set `app.py` as entrypoint.
4. Add secrets in the app settings.
5. Deploy.

## Notes / Limitations
- Free API tiers may rate-limit requests.
- Intraday coverage depends on provider support and limits.
- This project is educational and not financial advice.
