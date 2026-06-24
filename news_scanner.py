"""
news_scanner.py — Fast News Sentiment Pre-Filter
=================================================
Fetches recent news headlines for a ticker via yfinance and scores them
with VADER (Valence Aware Dictionary and sEntiment Reasoner).

Why VADER instead of FinBERT here?
  • VADER is CPU-only, rule-based, and runs in <1 ms per headline.
  • FinBERT (sentiment_ai.py) is a 400 MB transformer — saved for the
    deep AI audit (Layer 4) where we already know the ticker is worth auditing.
  • This module acts as a cheap Layer 2.5 pre-filter: kills obviously toxic
    news BEFORE we burn time on LSTM training, SEC filing analysis, etc.

Contract (what main.py expects):
    scan_news(ticker: str) -> dict
        {
            "is_passed":   bool   — True = news safe to proceed
            "score":       float  — compound score in [-1.0, +1.0]
            "headline_count": int
            "reason":      str
            "headlines":   list[str]
        }
"""

from __future__ import annotations

import yfinance as yf

# VADER ships inside nltk; download the lexicon once on first run.
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except LookupError:
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()

# ── Thresholds ────────────────────────────────────────────────────────────────
# Compound score runs from -1.0 (most negative) to +1.0 (most positive).
# We block a ticker if the average compound score falls below BLOCK_THRESHOLD.
# We flag (warn but don't block) if below WARN_THRESHOLD.
BLOCK_THRESHOLD = -0.25   # strong negative news → hard block
WARN_THRESHOLD  = -0.05   # slightly negative → note in reason but don't block
MAX_HEADLINES   = 8       # how many recent headlines to analyse per ticker


def scan_news(ticker: str, max_headlines: int = MAX_HEADLINES) -> dict:
    """
    Fetch recent headlines for `ticker` and return a pass/fail verdict.

    Fail-safe behaviour:
      • If yfinance returns no news  → PASS (don't penalise tickers with low
        media coverage; the FinBERT gate in Layer 4 will still run).
      • If yfinance raises an error  → FAIL (fail-closed: we can't confirm
        the news environment is safe, so we block).

    Parameters
    ----------
    ticker        : str  — e.g. "AAPL"
    max_headlines : int  — number of headlines to sample (default 8)

    Returns
    -------
    dict with keys:
        is_passed      : bool
        score          : float   — average VADER compound, rounded to 4 dp
        headline_count : int
        reason         : str
        headlines      : list[str]
    """
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news  # list of dicts with at least a 'title' key

        # ── No news at all ───────────────────────────────────────────────────
        if not news:
            return {
                "is_passed":      True,
                "score":          0.0,
                "headline_count": 0,
                "reason":         f"{ticker}: No recent news — proceeding (neutral).",
                "headlines":      [],
            }

        # ── Score each headline with VADER ───────────────────────────────────
        headlines = [
            item.get("content", {}).get("title", "") or item.get("title", "")
            for item in news[:max_headlines]
        ]
        headlines = [h for h in headlines if h]   # drop empty strings

        if not headlines:
            return {
                "is_passed":      True,
                "score":          0.0,
                "headline_count": 0,
                "reason":         f"{ticker}: Headlines had no text — proceeding (neutral).",
                "headlines":      [],
            }

        scores    = [_vader.polarity_scores(h)["compound"] for h in headlines]
        avg_score = round(sum(scores) / len(scores), 4)

        # ── Verdict ──────────────────────────────────────────────────────────
        if avg_score <= BLOCK_THRESHOLD:
            reason = (
                f"{ticker}: VADER avg={avg_score:.4f} ≤ block threshold {BLOCK_THRESHOLD} "
                f"across {len(headlines)} headline(s) — news environment too negative."
            )
            is_passed = False
        elif avg_score <= WARN_THRESHOLD:
            reason = (
                f"{ticker}: VADER avg={avg_score:.4f} (slightly negative, above hard block) "
                f"— proceeding with caution."
            )
            is_passed = True
        else:
            reason = (
                f"{ticker}: VADER avg={avg_score:.4f} across {len(headlines)} headline(s) "
                f"— news environment neutral/positive."
            )
            is_passed = True

        return {
            "is_passed":      is_passed,
            "score":          avg_score,
            "headline_count": len(headlines),
            "reason":         reason,
            "headlines":      headlines,
        }

    except Exception as exc:
        # Fail CLOSED: unknown error → cannot confirm safe news environment.
        return {
            "is_passed":      False,
            "score":          0.0,
            "headline_count": 0,
            "reason":         f"{ticker}: News scan error (fail-safe — trade blocked): {exc}",
            "headlines":      [],
        }


if __name__ == "__main__":
    import sys
    _ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result  = scan_news(_ticker)
    print(f"\n=== NEWS SCAN: {_ticker} ===")
    print(f"  Passed        : {result['is_passed']}")
    print(f"  VADER avg     : {result['score']}")
    print(f"  Headlines seen: {result['headline_count']}")
    print(f"  Reason        : {result['reason']}")
    print(f"  Headlines:")
    for h in result["headlines"]:
        print(f"    • {h}")
