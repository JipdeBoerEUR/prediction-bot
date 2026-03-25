# Prediction Bot: The Unified Trading Intelligence 🤖📈

A high-performance algorithmic trading system that fuses **Insider Intelligence**, **Topological Market Analysis**, and **Deep Learning** to execute high-conviction trades with institutional-grade risk management.

---

## 🏛️ System Architecture

The bot is structured as a **Unified Trading Daemon**, coordinating multiple intelligence layers before any capital is at risk.

### 1. The Gatekeepers (Risk & Regime)
- **Topological Regime Engine (`statarb/regime_engine.py`)**: Uses Persistent Homology (TDA) to detect market stress. If correlations collapse (indicating panic), the bot stops all new entries.
- **Sector Shield**: A dynamic exposure manager that prevents the bot from over-concentrating in any single industry (built into `main.py`).
- **Earnings Blackout**: Automatically vetoes trades within 7 days of an earnings release to avoid "gap" volatility.
- **Technical Eyes**: A 200-day SMA trend filter to ensure the bot isn't "catching a falling knife."

### 2. The Intelligence Layers (AI & NLP)
- **FinBERT Sentiment 🧠**: Uses a Transformer-based model optimized for finance to analyze the "tone" of recent news headlines.
- **SEC Auditor 📄**: Deep-scans 10-K and 10-Q filings for specific risk-factor keywords and sentiment shifts.
- **Earnings Analyst 🎙️**: Evaluates the specific language used by management during the most recent earnings calls.
- **LSTM Price Predictor 📈**: A Recurrent Neural Network that forecasts short-term (5-day) price movement based on recent OHLCV data.
- **Trade Learner 🎓**: A RandomForest model that looks at your own `trade_history.csv` to learn which insider-buy patterns actually result in winners, dynamically adjusting your Kelly Fraction.

### 3. The Execution Engine (Quant)
- **Insider Listener (`sec_listener.py`)**: Real-time monitoring of Open Market Purchases (Form 4).
- **Scoring Engine**: Weighs the "Conviction Score" based on the executive's role (CEO > 10% Owner).
- **Fractional Kelly Brakes (`kelly_size.py`)**: Mathematically optimizes position sizing to prevent ruin, capped at 5% of your total account.
- **Alpaca Execution**: Secure, automated order entry via the Alpaca Markets API.

---

## 🚀 Getting Started

If this is your first time setting up the bot, please follow the **[QUICKSTART.md](./QUICKSTART.md)** for a zero-to-hero installation guide.

### Core Commands
- **Run the Unified Daemon**: `python main.py` (Runs scans at 9:00 AM and 3:30 PM daily).
- **Manual Backtest**: `python advanced_backtest.py` (Simulate the entire AI stack on past data).
- **Run Tests**: `python -m pytest test_kelly_size.py`

---

## 💡 Philosophy
Most retail bots trade on "lagging" indicators (RSIs, MACDs). This bot trades on **Leading Indicators**: 
1. **Insider Motivation**: Why is the CEO buying?
2. **Market Topology**: Is the market broad or consolidating for a crash?
3. **AI Interpretation**: What is the subtle "tone" of the news and filings?

---

## 🔮 Roadmap
- **Topological Stat-Arb Integration**: Full automated pair-trading based on H1 persistence cycles.
- **Discord/Telegram Alerts**: Real-time push notifications for every AI evaluation.
- **Multi-Broker Support**: Integration for Interactive Brokers (IBKR) for lower margin rates.
