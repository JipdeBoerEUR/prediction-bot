# config.example.py
# ─────────────────────────────────────────────────────────────────────────────
# Copy this file to statarb/config.py and fill in your values.
# statarb/config.py is gitignored — never commit live credentials.
# ─────────────────────────────────────────────────────────────────────────────

# --- Live runner settings ---
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "JPM", "XOM", "CVX", "BAC"]

HISTORY_DAYS  = 365
LOOKBACK_DAYS = 22
SECTOR_ONLY   = True
GRAPH_THRESHOLD   = 0.30
REGULARIZATION    = 1e-6

USE_REGIME_FILTER         = True
REGIME_MIN_HEALTH         = 0.34
LIQUIDATE_ON_REGIME_BREACH = False

COST_BPS   = 0.08
QTY_SHARES = 10
PAPER      = True
DRY_RUN    = True
LOG_DIR    = "logs"
TIMEZONE   = "Europe/Amsterdam"

# --- Alpaca API Keys (always load from environment — never hardcode) ---
import os
from dotenv import load_dotenv
load_dotenv(override=True)
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# --- ML / Brain training settings ---
PROVIDER      = "alpaca"
ALPACA_FEED   = "iex"
DATASET_START = "2020-01-01"
DATASET_END   = None
INTERVAL      = "60m"       # 1-hour bars
BRAIN_MODEL_TYPE = "hgb"

RESIDUAL_N_JOBS        = -1
RESIDUAL_JOBLIB_BACKEND = "loky"
EVENT_N_JOBS           = -1
EVENT_JOBLIB_BACKEND   = "loky"

# --- Strategy parameters (Phase 3 Optuna-optimised defaults) ---
ENTRY_Z       = 1.74
HOLD_BARS     = 101
LOOKBACK_BARS = 985
EXEC_LAG_BARS = 1

EXIT_Z        = 0.31
EXIT_MIN_PROB = 0.28

MIN_PROB         = 0.32
MIN_PROB_MODE    = "quantile"
MIN_PROB_QUANTILE = 0.90

TRAIL_STOP_PCT = 0.012
RSI_PERIOD     = 14
RSI_MIN_LONG   = 30.0
RSI_MAX_SHORT  = 70.0
VIX_HIGH       = 25.0
VIX_LEVERAGE_MULT = 0.40
VIX_HALT       = 40.0
BEAR_LEVERAGE_MULT = 0.60

BUY_THRESH  = -0.01
SELL_THRESH =  0.03

# --- Risk / sizing ---
MAX_GROSS_LEVERAGE    = 2.0
ANNUAL_RISK_TARGET    = 0.25
MAX_WEIGHT_PER_ASSET  = 0.25
MIN_QTY               = 0.001
FORCE_FRACTIONAL_SHARES = True
STRENGTH_CAP          = 6.0
EDGE_POWER            = 1.2
SLIPPAGE_BPS          = 0.0
SIZING_METHOD         = "mean_variance"
COV_REGULARIZATION    = 1e-3

# --- Ticker universe (S&P 500) ---
# Full list lives in statarb/config.py. Add / remove tickers freely.
EQUITIES_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMD', 'GOOG', 'GOOGL', 'META',
    'AMZN', 'TSLA', 'JPM', 'BAC', 'WFC', 'V', 'MA',
    # ... see statarb/config.py for the full 435-ticker S&P 500 list
]

# ─────────────────────────────────────────────────────────────────────────────
# Topical Trading Engine — BERTopic + FinBERT narrative momentum
# ─────────────────────────────────────────────────────────────────────────────
TOPICAL_TRADING = {
    # BERTopic settings
    "min_topic_size":       3,      # min headlines to form a topic cluster
    "n_gram_range":         (1, 2), # unigrams + bigrams for topic labels
    "top_n_topics":         10,     # how many topics to surface per scan

    # Momentum scoring
    "momentum_window_hrs":  24,     # rolling window for headline velocity
    "min_momentum_score":   0.05,   # below this → topic too slow, skip
    "min_sentiment_score":  0.10,   # FinBERT net-positive threshold

    # Candidate ranking
    "min_exposure_score":   0.20,   # ticker must score ≥ this on topic keywords
    "max_candidates":       10,     # max tickers surfaced per topic per scan

    # RSS feed sources (all free, no API key required)
    "rss_feeds": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.ft.com/rss/home/us",
        "https://www.wsj.com/xml/rss/3_7085.xml",
        "https://seekingalpha.com/feed.xml",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
        "https://www.investors.com/category/news/rss/",
        "https://finance.yahoo.com/news/rssindex",
    ],
}
