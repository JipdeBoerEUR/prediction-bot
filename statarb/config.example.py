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

# --- Ticker universe (large-cap S&P 500 constituents) ---
# 306 liquid, well-known large/mega-cap names across all 11 GICS sectors.
#
# PROVENANCE / ACCURACY NOTE: compiled from general knowledge, not fetched
# from a live index-membership feed — I (the assistant that generated this)
# cannot verify current S&P 500 constituency against an authoritative source
# from this environment. Index membership changes over time (additions,
# removals, M&A, spinoffs). Treat this as "a broad, diversified, liquid
# large-cap universe good enough to build a meaningful correlation graph and
# run the strategy end-to-end" — not as a guaranteed-current, complete
# 500-member list. For exact current membership, cross-check against a
# public source (e.g. a maintained S&P 500 constituents CSV) before relying
# on this for anything beyond development/backtesting.
#
# Add / remove tickers freely — statarb's graph engine only needs a large,
# liquid, diversified set; exact index membership doesn't have to be perfect.
EQUITIES_TICKERS = [
    'AAPL', 'ABBV', 'ABNB', 'ADBE', 'AMAT', 'AMD', 'AMGN', 'AMZN',
    'AVGO', 'AXP', 'BA', 'BAC', 'BKNG', 'BLK', 'BMY', 'C',
    'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM',
    'CVS', 'CVX', 'DE', 'DHR', 'DIS', 'DUK', 'F', 'FCX',
    'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON',
    'INTC', 'ISRG', 'JNJ', 'JPM', 'KMI', 'KO', 'LIN', 'LLY',
    'LMT', 'LOW', 'LRCX', 'LULU', 'MA', 'MAR', 'MCD', 'META',
    'MO', 'MPC', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM',
    'NFLX', 'NKE', 'NOW', 'NVDA', 'ORCL', 'OXY', 'PANW', 'PEP',
    'PFE', 'PG', 'PLTR', 'PM', 'PNC', 'PSX', 'PYPL', 'QCOM',
    'REGN', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SNOW', 'SO',
    'T', 'TGT', 'TJX', 'TMO', 'TMUS', 'TSLA', 'UNH', 'UNP',
    'UPS', 'USB', 'V', 'VLO', 'VRTX', 'VZ', 'WFC', 'WMT',
    'XOM', 'TXN', 'IBM', 'CSCO', 'ACN', 'INTU', 'KLAC', 'SNPS',
    'CDNS', 'ADI', 'FTNT', 'ANSS', 'ROP', 'GLW', 'HPQ', 'HPE',
    'NTAP', 'STX', 'WDC', 'JNPR', 'TER', 'MCHP', 'ON', 'KEYS',
    'TYL', 'ZBRA', 'GDDY', 'AKAM', 'CDW', 'FFIV', 'EPAM', 'EA',
    'TTWO', 'WBD', 'OMC', 'IPG', 'MTCH', 'LYV', 'NWSA', 'FOX',
    'PARA', 'ORLY', 'ROST', 'YUM', 'CMG', 'AZO', 'HLT', 'DHI',
    'LEN', 'NVR', 'PHM', 'EBAY', 'ETSY', 'RL', 'GRMN', 'POOL',
    'APTV', 'BBY', 'DPZ', 'WYNN', 'MGM', 'CZR', 'RCL', 'CCL',
    'NCLH', 'EXPE', 'ULTA', 'TSCO', 'MDLZ', 'KHC', 'GIS', 'KMB',
    'STZ', 'HSY', 'MKC', 'CHD', 'CLX', 'SYY', 'KR', 'DG',
    'DLTR', 'ADM', 'TAP', 'TSN', 'EOG', 'WMB', 'OKE', 'HAL',
    'BKR', 'DVN', 'FANG', 'HES', 'CTRA', 'APA', 'MRO', 'SPGI',
    'CB', 'PGR', 'MMC', 'AON', 'TFC', 'BK', 'STT', 'MET',
    'AIG', 'PRU', 'AFL', 'TRV', 'ALL', 'FITB', 'HBAN', 'RF',
    'CFG', 'KEY', 'MTB', 'NTRS', 'SYF', 'DFS', 'ABT', 'SYK',
    'BSX', 'ZTS', 'BDX', 'HCA', 'IDXX', 'IQV', 'MRNA', 'WST',
    'DXCM', 'RMD', 'ALGN', 'CAH', 'MCK', 'COR', 'ELV', 'CI',
    'HUM', 'MDT', 'ADP', 'CSX', 'NSC', 'ITW', 'EMR', 'ETN',
    'PH', 'GD', 'NOC', 'WM', 'FDX', 'CMI', 'ROK', 'CTAS',
    'PCAR', 'IR', 'AME', 'URI', 'PWR', 'FAST', 'DAL', 'LUV',
    'UAL', 'AAL', 'APD', 'ECL', 'NUE', 'DOW', 'DD', 'PPG',
    'ALB', 'CTVA', 'LYB', 'IFF', 'PLD', 'AMT', 'EQIX', 'CCI',
    'PSA', 'O', 'DLR', 'SPG', 'VICI', 'AVB', 'EQR', 'VTR',
    'ESS', 'MAA', 'UDR', 'D', 'AEP', 'EXC', 'SRE', 'XEL',
    'ED', 'PEG', 'WEC', 'ES', 'FE', 'ETR', 'AEE', 'PPL',
    'CMS', 'DTE',
]

# --- Engine registry ------------------------------------------------------
# main.py hard-requires ENGINES["equities"] (tickers, model_path, and
# optional per-engine overrides of the strategy parameters above). Without
# this dict, main.py raises KeyError('ENGINES') on startup — everything below
# just wires the settings already defined above into the shape main.py and
# statarb/dataset_builder_v2.py expect. Add more engines (e.g. "forex") by
# adding more entries here.
ENGINES = {
    "equities": {
        "tickers":      EQUITIES_TICKERS,
        "model_path":   "models/brain_model_equities.joblib",
        # Correlation-graph warm-up window (live daily bars, main.py always
        # fetches interval="1d" for this regardless of INTERVAL below).
        "start":        "2023-01-01",
        "end":          None,
        # Live statarb exit threshold (main.py run_statarb_exits).
        "sell_thresh":  SELL_THRESH,
        # Historical dataset-builder settings (statarb/dataset_builder_v2.py).
        "interval":       INTERVAL,
        "entry_z":        ENTRY_Z,
        "hold_bars":      HOLD_BARS,
        "exec_lag_bars":  EXEC_LAG_BARS,
        "lookback_bars":  LOOKBACK_BARS,
        "cost_bps":       COST_BPS,
        "sector_only":    SECTOR_ONLY,
        "graph_threshold": GRAPH_THRESHOLD,
        "regularization": REGULARIZATION,
        "use_regime_filter":  USE_REGIME_FILTER,
        "regime_min_health":  REGIME_MIN_HEALTH,
        # Diffusion coefficient — no top-level ALPHA constant exists yet;
        # 0.8 is a reasonable starting point but was NOT tuned for this repo.
        # Re-run optimize_params.py and update this after the diffusion
        # operator fix (see CRITICAL_ISSUES_ANALYSIS.md / PR history) —
        # the old tuned value was fit against a different (buggy) operator.
        "alpha":          0.8,
    },
}

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
