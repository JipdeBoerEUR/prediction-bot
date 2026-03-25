# ⚡ Quickstart: Launching the Prediction Bot

Follow these steps to get your unified trading daemon up and running for the first time.

## 1. Environment Setup

### Prerequisites
- Python 3.10 or 3.11 (3.12+ may have giotto-tda compatibility issues)
- An **Alpaca Markets** account (Paper Trading keys are free)

### Clone & Install
```bash
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install core dependencies
pip install yfinance requests beautifulsoup4 pandas numpy scikit-learn joblib transformers torch schedule alpaca-trade-api pytest

# Install News Sentiment NLP
pip install vaderSentiment

# Optional: Install TDA Engine for Market Regime Detection
pip install giotto-tda
# OR (lighter fallback)
pip install gudhi
```

---

## 2. Configuration

### API Keys
The bot requires your Alpaca Paper Trading keys to operate. **Never share these or commit them to Git.**

Set them as environment variables in your terminal:
```powershell
$env:ALPACA_API_KEY="YOUR_KEY_HERE"
$env:ALPACA_SECRET_KEY="YOUR_SECRET_HERE"
```

### Portfolio Initialization
Create or verify `portfolio.json` in the root directory. This tells the bot how much capital it has and what it currently holds:
```json
{
  "account": {
    "buying_power": 100000.0
  },
  "holdings": []
}
```

---

## 3. First-Time Run (Initialization)

When you launch the bot for the first time, it will:
1. **Load the AI Models**: It will download the FinBERT model (~400MB) automatically.
2. **Train the Trade Learner**: It will read `trade_history.csv` to calibrate itself.
3. **Run Initial Scans**: It will immediately scan for insider buys and stat-arb opportunities.

**COMMAND:**
```bash
python main.py
```

---

## 4. "Forever" Mode (Daemon)

The bot is designed to run in a continuous loop (Daemon mode). Once launched:
- **09:00 AM**: Runs the **Insider Strategy** (Scanning for CEO buys).
- **09:30 AM**: Runs the **Stat-Arb Strategy** (Topological pairs scan).
- **03:30 PM**: Runs **Portfolio Monitor** (Adjusting stops and trimming winners).

### How to Manage the Daemon
- **Log Monitoring**: All evaluations are saved to `logger.py`'s output and `trade_history.csv`.
- **Stopping**: Press `Ctrl+C` in the terminal to gracefully exit. The bot will finish its current task before closing.
- **Production Tip**: Use a process manager like **PM2** or a simple **Windows Task Scheduler** to ensure the script restarts if your computer reboots.

---

## 5. Verification
To ensure everything is working correctly, run a smoke test on the AI forecasting stack:
```bash
python -c "from ai.price_predictor import predict_price_movement; print(predict_price_movement('AAPL'))"
```
If you see a JSON response with a "confidence" score, your AI engines are live!
