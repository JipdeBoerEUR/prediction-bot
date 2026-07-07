# Prediction Bot — Topical Trading Engine

> **An end-to-end autonomous trading system combining statistical arbitrage with NLP-driven narrative momentum.** Built as a portfolio project for quantitative finance applications.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Alpaca](https://img.shields.io/badge/Broker-Alpaca%20Paper-yellow)](https://alpaca.markets)

---

## Overview

This bot runs a dual-source alpha pipeline: one scout discovers **topical momentum** from live financial news using BERTopic + FinBERT, and a second scout identifies **statistical arbitrage** opportunities from mean-reversion signals on correlated equity pairs. Both feeds merge into a unified candidate pool that is filtered through a multi-gate AI audit before execution.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  OrchestratorContext                │
│          (macro regime + market session)             │
└────────────────┬──────────────┬──────────────────────┘
                 │              │
    ┌────────────▼───┐  ┌───────▼──────────┐
    │  Scout A       │  │  Scout B         │
    │  Topic Engine  │  │  StatArb Engine  │
    │  (BERTopic +   │  │  (Graph-based    │
    │   FinBERT NLP) │  │   cointegration) │
    └────────────┬───┘  └───────┬──────────┘
                 └──────┬───────┘
              ┌─────────▼──────────┐
              │  UnifiedCandidate  │
              │  Pool (deduped)    │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────────────────────────┐
              │  Multi-Gate AI Audit                   │
              │  Gate 1: Sentiment pre-filter           │
              │  Gate 2: FinBERT re-score               │
              │  Gate 3: LSTM directional forecast      │
              │  Gate 4: RAG risk report (Gemini)       │
              └─────────┬──────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  Kelly Sizing +    │
              │  Risk Manager      │
              │  (VIX gating,      │
              │   regime filter)   │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  Alpaca Execution  │
              │  (Paper / Live)    │
              └────────────────────┘
```

---

## Key Components

### 1. Topical Trading Engine (`ai/topic_engine.py`)
- Aggregates headlines from **10 free RSS feeds** (Reuters, Bloomberg, CNBC, FT, WSJ, etc.)
- Clusters headlines into topics using **BERTopic** (UMAP dimensionality reduction + HDBSCAN clustering)
- Scores each topic cluster with **FinBERT** (ProsusAI/finbert) for financial sentiment
- Maps high-momentum topics to S&P 500 tickers via keyword exposure scoring
- Outputs: `(topic_label, momentum_score, sentiment_score, tickers[])`

### 2. Statistical Arbitrage Engine (`statarb/`)
- Builds a **correlation graph** across the S&P 500 (~435 tickers)
- Identifies cointegrated pairs and computes Ornstein–Uhlenbeck residuals
- **Brain Engine**: Gradient-boosted classifier (`HistGradientBoosting`) trained on historical spread features to predict mean-reversion probability
- **Regime Engine**: Filters out signals during adverse macro regimes (VIX spikes, bear phases)
- Parameters optimised with **Optuna** (Bayesian hyperparameter search)

### 3. AI Audit Pipeline (`ai/`)
| Module | Role |
|---|---|
| `sentiment_ai.py` | FinBERT sentiment scoring on live news |
| `price_predictor.py` | LSTM directional forecast (bullish/bearish) |
| `sec_analyzer.py` | 10-K/10-Q regulatory risk parsing |
| `earnings_analyzer.py` | Earnings call transcript tone analysis |
| `rag_library.py` | Retrieval-augmented risk report via Gemini |
| `news_scanner.py` | VADER fast pre-filter (< 1ms per ticker) |
| `trade_learner.py` | Online learning from trade outcomes |

### 4. Risk & Execution
- **Kelly Criterion** position sizing with drawdown cap
- **Mean-variance optimisation** for portfolio weights (with covariance regularisation)
- Macro filter: VIX halt at 40, leverage cut at 25
- Trailing stop-loss per position
- Paper trading via **Alpaca Markets API**

---

## Ticker Universe

Full **S&P 500** (~435 tickers) across all 11 GICS sectors:
- Information Technology, Communication Services, Consumer Discretionary, Consumer Staples
- Health Care, Financials, Industrials, Energy, Materials, Real Estate, Utilities

---

## Tech Stack

| Category | Libraries |
|---|---|
| NLP / Topic Modelling | BERTopic, sentence-transformers, UMAP, HDBSCAN |
| Financial NLP | ProsusAI/FinBERT (HuggingFace Transformers) |
| Deep Learning | PyTorch, Transformers |
| Classical ML | scikit-learn (HistGradientBoosting), Optuna |
| Market Data | yfinance, Alpaca Markets API |
| RAG / LLM | Google Gemini (google-generativeai), ChromaDB |
| Data | pandas, numpy, scipy |
| Scheduling | APScheduler |
| Broker | alpaca-py (paper + live) |
| Containerisation | Docker |

---

## Setup

### 1. Clone & install
```bash
git clone https://github.com/Arrowpowercod/prediction-bot.git
cd prediction-bot
pip install -r requirements.txt
```

### 2. Configure
```bash
# Copy the example config
cp statarb/config.example.py statarb/config.py

# Create your .env (never committed to git)
cat > .env << EOF
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
GOOGLE_API_KEY=your_gemini_key       # optional — for RAG risk reports
TELEGRAM_BOT_TOKEN=your_bot_token   # optional — for trade alerts
TELEGRAM_CHAT_ID=your_chat_id       # optional
EOF
```

Get a free Alpaca paper trading account at [alpaca.markets](https://alpaca.markets).  
Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### 3. Run
```bash
# Paper trading (dry run)
python main.py

# Or via Docker
docker build -t prediction-bot .
docker run --env-file .env prediction-bot
```

### 4. (Optional) Retrain the Brain Model
After setup, retrain the stat-arb classifier on the full S&P 500 universe:
```bash
python statarb/dataset_builder_v2.py
```
This downloads ~5 years of hourly OHLCV data and trains the gradient-boosted classifier. Takes 15–30 min on first run; cached thereafter.

---

## Backtesting

### Walk-forward simulation of the real pipeline

```bash
python backtest_walkforward.py                    # 2019 → today, vol-scaled exits
python backtest_walkforward.py --exits fixed      # A/B: legacy fixed -7%/+15% exits
python backtest_walkforward.py --dollar-neutral   # add the short book
```

`backtest_walkforward.py` replays the **actual production signal path** day by
day: trailing-window correlation graph → Laplacian diffusion residuals (the
same `statarb.sim_engine` solver the live engine uses) → residual z-score
entries filled at the *next bar's open* → exits through the *same*
`trade_utils.check_position_exit` function the live monitor calls. Regime
gate (SPY 200-DMA / VIX), per-side transaction costs, and intraday High/Low
stop checks with pessimistic ordering are all modeled. Outputs
`BACKTEST_REPORT.md` (in-sample vs out-of-sample vs SPY metrics table) and
`backtest_report.png` (equity + drawdown chart). Price data is cached in
`bt_cache/` so reruns are instant.

Honest scope note: the topic (news momentum) sleeve is **not** simulated —
no historical archive of the RSS headline stream exists, and synthesizing one
would be fiction. The report says so explicitly.

### Legacy event study

```bash
python backtest.py
```

The original event-study backtester replays historical signals through the regime filter, risk sizing and execution costs and reports Sharpe, max drawdown, win rate and a P&L curve.

---

## Exit Engineering

Exits are where most of the expectancy lives, so they are volatility-aware
and unit-tested (`tests/test_exit_logic.py`):

- **σ-scaled stops** — the stop distance is `3×σ_daily` (clamped 4–15%),
  computed per position at entry. A fixed −7% stop is ~3σ for a staples stock
  but ~1σ for a high-beta chip name; scaling normalizes exit behavior across
  the universe. (`trade_utils.vol_scaled_stop_pct`)
- **Trailing stops instead of profit caps** — winners run until they give
  back one stop-width from their peak (high-water mark persisted in
  `bot_positions.json`, so it survives restarts) rather than being amputated
  at +15%.
- **Time decay exits** — a topic-momentum position that has gone nowhere in
  10 trading days is a dead thesis occupying a scarce position slot; it gets
  recycled.
- **Earnings blackout** — no entries within 3 days of a known earnings date:
  a 5-day directional forecast into an earnings gap is a coin flip, not a
  signal.
- **Strategy-aware exit routing** — a persistent position ledger records
  which strategy opened each position, so the statarb mean-reversion exit
  never force-sells a topic-momentum winner, and shorts are covered on
  residual reversal.

The same exit code runs in the live monitor and the walk-forward backtester,
so the A/B flag (`--exits vol|fixed`) measures exactly what production would do.

---

## Strategy Logic

### Topical Momentum Alpha
1. Every scan cycle, fetch headlines from 10 RSS feeds
2. Cluster into topics with BERTopic (minimum 3 headlines per cluster)
3. Score each topic cluster with FinBERT — positive sentiment + high velocity = momentum signal
4. Map topic to tickers with `>= 0.20` keyword exposure score
5. Send top tickers through the 4-gate AI audit
6. Size with Kelly Criterion, execute via Alpaca

### Statistical Arbitrage Alpha
1. Build correlation graph on S&P 500 (threshold: |ρ| > 0.30)
2. Compute Ornstein–Uhlenbeck residuals for correlated pairs
3. Brain Engine predicts mean-reversion probability
4. Enter when `z-score < -1.74` and `p(reversion) > 0.32`
5. Exit when `z-score > 0.31` or trailing stop hit

---

## Project Structure

```
prediction-bot/
├── main.py                    # Orchestrator — dual-scout pipeline
├── ai/
│   ├── news_aggregator.py     # RSS headline fetcher (10 feeds, 15-min cache)
│   ├── topic_engine.py        # BERTopic + FinBERT topic discovery
│   ├── topic_exposure.py      # Ticker-to-topic exposure scoring
│   ├── sentiment_ai.py        # FinBERT news sentiment
│   ├── price_predictor.py     # LSTM directional forecast
│   ├── sec_analyzer.py        # SEC filing risk analysis
│   ├── earnings_analyzer.py   # Earnings transcript analysis
│   ├── rag_library.py         # RAG risk report (Gemini + ChromaDB)
│   └── trade_learner.py       # Online learning from outcomes
├── statarb/
│   ├── config.example.py      # Configuration template
│   ├── brain_engine.py        # Gradient-boosted signal classifier
│   ├── graph_engine.py        # Correlation graph builder
│   ├── signal_engine.py       # Entry/exit signal generation
│   ├── regime_engine.py       # Macro regime filter
│   ├── risk_manager.py        # Position sizing + leverage control
│   ├── dataset_builder_v2.py  # Historical data + feature pipeline
│   └── sim_engine.py          # Simulation engine
├── backtest.py                # Historical simulation + metrics
├── news_scanner.py            # VADER fast pre-filter
├── notifier.py                # Telegram / email alerts
├── macro_filter.py            # VIX + regime gating
├── kelly_size.py              # Kelly Criterion sizing
├── logger.py                  # Structured trade logging
├── Dockerfile                 # Container definition
└── requirements.txt           # Python dependencies
```

---

## Disclaimer

This project is for **educational and portfolio demonstration purposes only**. It runs in paper trading mode by default. Past performance of any backtested strategy does not guarantee future results. This is not financial advice.

---

## Author

Built by [@Arrowpowercod](https://github.com/Arrowpowercod) — working student quant finance candidate.
