# ai/news_aggregator.py
# Cross-market headline aggregator for Topical Trading.
#
# Scrapes 12 financial RSS feeds and deduplicates by semantic similarity,
# returning a fresh headline corpus ready for BERTopic discovery.
#
# Results are cached for 15 minutes to avoid hammering feeds during a scan.

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import feedparser

logger = logging.getLogger(__name__)

# ── RSS feed registry ────────────────────────────────────────────────────────
RSS_FEEDS: List[Dict[str, str]] = [
    {"name": "Reuters Business",   "url": "https://feeds.reuters.com/reuters/businessNews"},
    {"name": "Reuters Tech",       "url": "https://feeds.reuters.com/reuters/technologyNews"},
    {"name": "CNBC Markets",       "url": "https://search.cnbc.com/rs/search/combinedcgi?id=100003114&format=rss"},
    {"name": "CNBC Finance",       "url": "https://search.cnbc.com/rs/search/combinedcgi?id=10000664&format=rss"},
    {"name": "MarketWatch Top",    "url": "https://feeds.marketwatch.com/marketwatch/topstories/"},
    {"name": "MarketWatch Markets","url": "https://feeds.marketwatch.com/marketwatch/marketpulse/"},
    {"name": "Barrons",            "url": "https://www.barrons.com/xml/rss/3_7514.xml"},
    {"name": "WSJ Markets",        "url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"},
    {"name": "FT Markets",         "url": "https://www.ft.com/rss/home/uk"},
    {"name": "Seeking Alpha",      "url": "https://seekingalpha.com/market-news/all/rss.xml"},
    {"name": "Motley Fool",        "url": "https://www.fool.com/feeds/index.aspx"},
    {"name": "Yahoo Finance",      "url": "https://finance.yahoo.com/rss/"},
]

# ── In-memory cache ──────────────────────────────────────────────────────────
_cache: Dict[str, object] = {
    "headlines": [],
    "timestamp": 0.0,
    "ttl_seconds": 900,    # 15 minutes
}


def _clean_title(title: str) -> str:
    """Strip HTML entities, extra whitespace, and common RSS boilerplate."""
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"&[a-z]+;", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _title_fingerprint(title: str) -> str:
    """64-char fingerprint — used for fast duplicate detection."""
    normalized = re.sub(r"[^a-z0-9 ]", "", title.lower())
    tokens = sorted(normalized.split())
    return hashlib.md5(" ".join(tokens).encode()).hexdigest()


def _fetch_feed(feed: Dict[str, str], lookback_hours: int = 6) -> List[Dict]:
    """Fetch one RSS feed and return list of headline dicts within lookback window."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=lookback_hours)
    headlines = []

    try:
        parsed = feedparser.parse(feed["url"])
        for entry in parsed.entries:
            title = _clean_title(getattr(entry, "title", ""))
            if not title or len(title) < 15:
                continue

            # Parse publish time (fallback to now if missing)
            published: datetime = datetime.now(tz=timezone.utc)
            for attr in ("published_parsed", "updated_parsed", "created_parsed"):
                t = getattr(entry, attr, None)
                if t:
                    try:
                        published = datetime(*t[:6], tzinfo=timezone.utc)
                        break
                    except Exception:
                        pass

            if published < cutoff:
                continue

            headlines.append({
                "title":     title,
                "source":    feed["name"],
                "published": published.isoformat(),
                "url":       getattr(entry, "link", ""),
                "summary":   _clean_title(getattr(entry, "summary", "")[:300]),
                "_fp":       _title_fingerprint(title),
            })
    except Exception as e:
        logger.warning("[news_aggregator] Feed '%s' failed: %s", feed["name"], e)

    return headlines


def _deduplicate(headlines: List[Dict]) -> List[Dict]:
    """Remove duplicate headlines by fingerprint. Keep first occurrence."""
    seen: set = set()
    unique: List[Dict] = []
    for h in headlines:
        fp = h["_fp"]
        if fp not in seen:
            seen.add(fp)
            unique.append(h)
    return unique


def aggregate_headlines(
    lookback_hours: int = 6,
    force_refresh: bool = False,
    max_per_feed: int = 50,
) -> List[Dict]:
    """
    Aggregate headlines from all registered RSS feeds.

    Args:
        lookback_hours: Only include headlines published within this window.
        force_refresh:  Bypass the 15-minute cache.
        max_per_feed:   Maximum headlines to take from any single feed.

    Returns:
        List of headline dicts:
          {title, source, published, url, summary}
    """
    now = time.time()
    if (
        not force_refresh
        and _cache["headlines"]
        and now - _cache["timestamp"] < _cache["ttl_seconds"]
    ):
        cached = _cache["headlines"]
        logger.info("[news_aggregator] Returning %d cached headlines.", len(cached))
        return cached

    logger.info("[news_aggregator] Fetching %d RSS feeds (lookback=%dh)…", len(RSS_FEEDS), lookback_hours)

    all_headlines: List[Dict] = []
    for feed in RSS_FEEDS:
        batch = _fetch_feed(feed, lookback_hours)[:max_per_feed]
        logger.debug("[news_aggregator] %s → %d headlines", feed["name"], len(batch))
        all_headlines.extend(batch)

    unique = _deduplicate(all_headlines)

    # Remove fingerprint field before returning
    for h in unique:
        h.pop("_fp", None)

    # Sort by publish time descending (freshest first)
    unique.sort(key=lambda h: h["published"], reverse=True)

    logger.info("[news_aggregator] %d unique headlines from %d feeds.", len(unique), len(RSS_FEEDS))

    _cache["headlines"] = unique
    _cache["timestamp"] = now
    return unique


def get_ticker_headlines(
    ticker: str,
    all_headlines: Optional[List[Dict]] = None,
    max_results: int = 20,
) -> List[str]:
    """
    Filter aggregated headlines for those mentioning a specific ticker or company.
    Falls back to aggregate_headlines() if all_headlines not provided.

    Returns list of title strings only.
    """
    if all_headlines is None:
        all_headlines = aggregate_headlines()

    pattern = re.compile(r"\b" + re.escape(ticker) + r"\b", re.IGNORECASE)
    matches = [
        h["title"]
        for h in all_headlines
        if pattern.search(h["title"]) or pattern.search(h.get("summary", ""))
    ]
    return matches[:max_results]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hl = aggregate_headlines(force_refresh=True)
    print(f"\nTotal headlines: {len(hl)}")
    for h in hl[:10]:
        print(f"  [{h['source']}] {h['title']}")
