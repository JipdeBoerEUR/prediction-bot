# ai/topic_exposure.py
# Ticker → Topic Exposure Mapper for Topical Trading.
#
# Pre-builds a dictionary mapping each ticker to a list of thematic
# keywords/topics it is exposed to. Used by topic_engine.py to identify
# which tickers are relevant to a discovered news topic.
#
# Exposure data is built from two sources (in priority order):
#   1. yfinance info (sector, industry, long business summary)
#   2. A hardcoded seed table for well-known names (instant, no API call)
#
# The full map is cached in memory and refreshed weekly.

from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Seed topic map for well-known tickers ─────────────────────────────────────
# Keywords here are matched against BERTopic cluster labels.
_SEED_MAP: Dict[str, List[str]] = {
    # Mega-cap Tech
    "AAPL":  ["iphone", "apple", "ios", "mac", "wearables", "app store", "services revenue",
               "consumer electronics", "smartphone", "vision pro"],
    "MSFT":  ["azure", "microsoft", "cloud computing", "copilot", "openai", "office 365",
               "enterprise software", "gaming", "activision", "ai infrastructure"],
    "NVDA":  ["nvidia", "gpu", "ai chips", "data center", "cuda", "semiconductor",
               "artificial intelligence hardware", "blackwell", "hopper", "h100", "b200"],
    "GOOG":  ["google", "alphabet", "search", "youtube", "gemini", "waymo",
               "digital advertising", "cloud ai", "android"],
    "GOOGL": ["google", "alphabet", "search", "youtube", "gemini", "waymo",
               "digital advertising", "cloud ai", "android"],
    "META":  ["meta", "facebook", "instagram", "whatsapp", "llama", "metaverse",
               "digital advertising", "social media", "reels", "threads"],
    "AMZN":  ["amazon", "aws", "e-commerce", "prime", "logistics", "alexa",
               "cloud computing", "retail", "advertising"],
    "TSLA":  ["tesla", "electric vehicle", "ev", "autonomous driving", "fsd",
               "energy storage", "battery", "gigafactory", "cybertruck", "robotaxi"],

    # Semiconductors
    "AMD":   ["amd", "cpu", "gpu", "semiconductor", "mi300", "ryzen", "epyc",
               "data center chips", "gaming chips", "ai accelerators"],
    "INTC":  ["intel", "cpu", "foundry", "x86", "semiconductor fabrication",
               "gaudi", "arc gpu", "panther lake"],
    "AVGO":  ["broadcom", "networking chips", "vmware", "asic", "wireless",
               "semiconductor", "ai networking"],
    "QCOM":  ["qualcomm", "snapdragon", "5g", "mobile chips", "automotive chips",
               "iot", "semiconductor"],
    "MU":    ["micron", "dram", "nand", "memory chips", "hbm", "semiconductor",
               "ai memory"],
    "AMAT":  ["applied materials", "semiconductor equipment", "chip fab", "deposition"],
    "LRCX":  ["lam research", "semiconductor equipment", "etch", "chip manufacturing"],

    # Software / Cloud
    "CRM":   ["salesforce", "crm", "enterprise software", "agentforce", "ai agents",
               "saas", "sales automation"],
    "ADBE":  ["adobe", "creative cloud", "firefly", "ai content generation",
               "digital media", "saas"],
    "NOW":   ["servicenow", "it automation", "enterprise workflow", "ai platform",
               "saas", "digital transformation"],
    "SNOW":  ["snowflake", "data cloud", "analytics", "saas", "data warehouse",
               "ai data platform"],
    "PLTR":  ["palantir", "ai software", "government contracts", "defense tech",
               "data analytics", "aip"],
    "ORCL":  ["oracle", "database", "cloud", "enterprise software", "saas",
               "ai infrastructure"],

    # Cybersecurity
    "PANW":  ["palo alto networks", "cybersecurity", "zero trust", "siem",
               "network security", "firewall"],

    # Fintech / Finance
    "V":     ["visa", "payments", "fintech", "credit card", "transaction volume",
               "consumer spending"],
    "MA":    ["mastercard", "payments", "fintech", "credit card", "consumer spending"],
    "PYPL":  ["paypal", "digital payments", "venmo", "fintech", "e-commerce payments"],
    "JPM":   ["jpmorgan", "banking", "interest rates", "fed policy", "investment banking",
               "credit markets", "financial regulation"],
    "BAC":   ["bank of america", "banking", "interest rates", "credit", "retail banking",
               "financial regulation"],
    "GS":    ["goldman sachs", "investment banking", "trading", "mergers acquisitions",
               "ipo", "credit markets"],
    "MS":    ["morgan stanley", "investment banking", "wealth management", "trading"],
    "WFC":   ["wells fargo", "banking", "mortgage", "interest rates", "consumer lending"],
    "BLK":   ["blackrock", "asset management", "etf", "passive investing", "rates"],
    "AXP":   ["american express", "credit card", "consumer spending", "fintech"],
    "SCHW":  ["charles schwab", "brokerage", "interest rates", "retail investing"],
    "COF":   ["capital one", "credit card", "consumer credit", "lending"],
    "USB":   ["us bancorp", "banking", "interest rates", "regional bank"],
    "PNC":   ["pnc", "banking", "interest rates", "regional bank"],
    "C":     ["citigroup", "banking", "global markets", "interest rates", "credit"],

    # Healthcare / Biotech
    "JNJ":   ["johnson johnson", "pharmaceutical", "medical devices", "drug approval"],
    "UNH":   ["unitedhealth", "health insurance", "medicare", "managed care", "payer"],
    "PFE":   ["pfizer", "drug", "pharmaceutical", "fda approval", "clinical trial"],
    "LLY":   ["eli lilly", "glp-1", "wegovy", "ozempic competitor", "diabetes drug",
               "obesity drug", "pharmaceutical", "fda approval"],
    "MRK":   ["merck", "keytruda", "oncology", "pharmaceutical", "drug approval"],
    "ABBV":  ["abbvie", "humira", "pharmaceutical", "immunology", "biotech"],
    "AMGN":  ["amgen", "biotech", "biologic drugs", "inflammation", "oncology"],
    "GILD":  ["gilead", "hiv", "antiviral", "biotech", "oncology"],
    "ISRG":  ["intuitive surgical", "robotic surgery", "da vinci", "medical devices"],
    "VRTX":  ["vertex", "cystic fibrosis", "rare disease", "biotech", "fda approval"],
    "REGN":  ["regeneron", "dupixent", "biotech", "biologic", "immunology"],
    "BMY":   ["bristol myers squibb", "oncology", "pharmaceutical", "immunology"],
    "TMO":   ["thermo fisher", "life sciences tools", "diagnostics", "biotech"],
    "DHR":   ["danaher", "life sciences", "diagnostics", "biotech tools"],
    "CVS":   ["cvs health", "pharmacy", "health insurance", "retail health"],

    # Consumer / Retail
    "WMT":  ["walmart", "retail", "consumer spending", "e-commerce", "grocery"],
    # NOTE: AMZN is defined above in the Tech section with a fuller keyword set.
    "COST": ["costco", "retail", "wholesale", "consumer spending", "membership"],
    "TGT":  ["target", "retail", "consumer spending", "discretionary"],
    "HD":   ["home depot", "housing market", "home improvement", "construction"],
    "LOW":  ["lowes", "housing market", "home improvement", "construction"],
    "MCD":  ["mcdonalds", "fast food", "consumer spending", "restaurants"],
    "NKE":  ["nike", "sportswear", "consumer", "china sales", "apparel"],
    "SBUX": ["starbucks", "coffee", "consumer spending", "china", "restaurants"],
    "TJX":  ["tjx", "off-price retail", "consumer", "discount retail"],
    "LULU": ["lululemon", "athleisure", "consumer", "apparel"],
    "MAR":  ["marriott", "hotels", "travel", "hospitality"],
    "BKNG": ["booking", "travel", "online travel", "hospitality"],
    "ABNB": ["airbnb", "travel", "short-term rental", "hospitality"],

    # Energy
    "XOM":  ["exxon", "oil", "natural gas", "energy prices", "opec", "refining"],
    "CVX":  ["chevron", "oil", "natural gas", "energy prices", "opec"],
    "COP":  ["conocophillips", "oil exploration", "energy", "lng"],
    "SLB":  ["schlumberger", "oilfield services", "drilling"],
    "OXY":  ["occidental", "oil", "carbon capture", "energy"],
    "MPC":  ["marathon petroleum", "refining", "oil", "energy"],
    "PSX":  ["phillips 66", "refining", "oil", "midstream"],
    "VLO":  ["valero", "refining", "ethanol", "energy"],
    "KMI":  ["kinder morgan", "natural gas pipelines", "midstream", "energy"],

    # Industrials / Defense
    "BA":   ["boeing", "aerospace", "aviation", "defense", "aircraft", "737 max"],
    "LMT":  ["lockheed martin", "defense", "f-35", "missiles", "government contracts"],
    "RTX":  ["raytheon", "defense", "missiles", "government contracts", "aerospace"],
    "CAT":  ["caterpillar", "construction", "mining", "infrastructure", "industrials"],
    "GE":   ["ge", "aerospace", "jet engines", "power generation", "industrials"],
    "UPS":  ["ups", "logistics", "e-commerce delivery", "package volume"],
    "HON":  ["honeywell", "industrials", "automation", "aerospace systems"],
    "DE":   ["deere", "agriculture", "farming equipment", "autonomous machinery"],
    "UNP":  ["union pacific", "railroad", "freight", "supply chain"],

    # EV / Clean Energy
    "NEE":  ["nextera", "solar", "wind", "clean energy", "utilities", "renewable"],
    "DUK":  ["duke energy", "utilities", "electric grid", "clean energy"],
    "SO":   ["southern company", "utilities", "nuclear", "clean energy"],
    "F":    ["ford", "electric vehicle", "ev", "f-150 lightning", "auto"],
    "GM":   ["general motors", "electric vehicle", "ev", "ultium", "auto"],

    # Materials
    "LIN":  ["linde", "industrial gases", "clean hydrogen", "chemicals"],
    "SHW":  ["sherwin-williams", "paint", "housing market", "construction"],
    "FCX":  ["freeport", "copper", "mining", "ev metals", "data center metals"],
    "NEM":  ["newmont", "gold", "mining", "precious metals", "inflation hedge"],

    # Media / Entertainment
    "DIS":  ["disney", "streaming", "espn", "theme parks", "box office", "hulu"],
    "NFLX": ["netflix", "streaming", "subscriber growth", "content", "ad tier"],
    "CMCSA":["comcast", "cable", "nbc", "streaming", "broadband"],
    "TMUS": ["t-mobile", "5g", "wireless", "telecom", "subscriber growth"],
    "VZ":   ["verizon", "telecom", "5g", "wireless", "broadband"],
    "T":    ["at&t", "telecom", "5g", "wireless", "broadband", "fiber"],
    "CHTR": ["charter", "cable", "broadband", "internet", "spectrum"],

    # Consumer Staples
    "PG":   ["procter gamble", "consumer staples", "household products", "inflation"],
    "KO":   ["coca-cola", "beverages", "consumer staples", "pricing power"],
    "PEP":  ["pepsico", "beverages", "snacks", "consumer staples", "pricing power"],
    "PM":   ["philip morris", "tobacco", "iqos", "nicotine", "heated tobacco"],
    "MO":   ["altria", "tobacco", "vaping", "nicotine"],
    "CL":   ["colgate", "oral care", "consumer staples", "household products"],
}


# ── In-memory cache ──────────────────────────────────────────────────────────
_exposure_cache: Dict[str, List[str]] = {}
_cache_ts: float = 0.0
_CACHE_TTL: float = 7 * 24 * 3600   # 1 week


def _enrich_from_yfinance(ticker: str, seed_keywords: List[str]) -> List[str]:
    """
    Try to extract additional keywords from yfinance info dict
    (sector, industry, longBusinessSummary). Falls back silently.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}

        extras: List[str] = list(seed_keywords)
        for field in ("sector", "industry"):
            val = info.get(field, "")
            if val:
                extras.append(val.lower())

        summary = info.get("longBusinessSummary", "") or ""
        # Extract noun phrases from the first two sentences
        sentences = summary.split(".")[:2]
        excerpt = " ".join(sentences).lower()
        # Grab multi-word phrases that could be meaningful topics
        words = re.findall(r"[a-z]{4,}", excerpt)
        stop = {"with", "that", "this", "from", "they", "their", "have",
                "been", "will", "also", "into", "which", "company",
                "provides", "through", "various", "products", "services"}
        extras += [w for w in words if w not in stop][:10]
        return list(dict.fromkeys(extras))   # deduplicate, preserve order

    except Exception:
        return seed_keywords


def build_exposure_map(
    tickers: List[str],
    use_yfinance: bool = False,
) -> Dict[str, List[str]]:
    """
    Build or return the cached ticker → topic-keyword exposure map.

    Args:
        tickers:      List of tickers to include (uses SEED_MAP for known ones,
                      yfinance fallback for unknown ones if use_yfinance=True).
        use_yfinance: Whether to enrich with yfinance info (slower but richer).

    Returns:
        Dict mapping ticker → list of lowercase keyword strings.
    """
    global _exposure_cache, _cache_ts

    now = time.time()
    if _exposure_cache and now - _cache_ts < _CACHE_TTL:
        # The cache is keyed by whichever ticker list built it first. Returning
        # .get(t, []) meant a later call with extra tickers silently got zero
        # exposure for them (for up to a week). Build just the missing ones
        # and merge instead.
        missing = [t for t in tickers if t not in _exposure_cache]
        for ticker in missing:
            seed = _SEED_MAP.get(ticker, [ticker.lower()])
            _exposure_cache[ticker] = (
                _enrich_from_yfinance(ticker, seed) if use_yfinance else seed
            )
        if missing:
            logger.info("[topic_exposure] Added %d missing tickers to cache.", len(missing))
        return {t: _exposure_cache[t] for t in tickers}

    logger.info("[topic_exposure] Building exposure map for %d tickers…", len(tickers))
    result: Dict[str, List[str]] = {}

    for ticker in tickers:
        seed = _SEED_MAP.get(ticker, [ticker.lower()])
        if use_yfinance:
            result[ticker] = _enrich_from_yfinance(ticker, seed)
        else:
            result[ticker] = seed

    _exposure_cache = result
    _cache_ts = now
    logger.info("[topic_exposure] Exposure map built for %d tickers.", len(result))
    return result


def score_ticker_exposure(
    ticker: str,
    topic_label: str,
    topic_words: List[str],
    exposure_map: Optional[Dict[str, List[str]]] = None,
) -> float:
    """
    Score how strongly a ticker is exposed to a BERTopic cluster.

    Matching logic:
      - Exact keyword match in topic_words → 1.0 per match
      - Substring match in topic_label    → 0.5 per match
      - Score normalised to [0, 1]

    Args:
        ticker:       The stock ticker.
        topic_label:  Human-readable topic label from BERTopic (e.g. "AI chip shortage").
        topic_words:  Top representative words from the BERTopic cluster.
        exposure_map: Pre-built map. Built on-the-fly if None.

    Returns:
        Exposure score in [0.0, 1.0].
    """
    if exposure_map is None:
        exposure_map = build_exposure_map([ticker])

    keywords = [kw.lower() for kw in exposure_map.get(ticker, [])]
    if not keywords:
        return 0.0

    topic_label_lower = topic_label.lower()
    topic_words_lower = [w.lower() for w in topic_words]

    score = 0.0
    for kw in keywords:
        # Exact word match in topic representative words
        for tw in topic_words_lower:
            if kw in tw or tw in kw:
                score += 1.0
                break
        # Substring match in topic label
        if kw in topic_label_lower:
            score += 0.5

    # Normalise: cap at max possible and scale to [0, 1]
    max_possible = len(keywords) * 1.5
    return min(score / max_possible, 1.0) if max_possible > 0 else 0.0


if __name__ == "__main__":
    from statarb.config import EQUITIES_TICKERS
    emap = build_exposure_map(EQUITIES_TICKERS, use_yfinance=False)
    print(f"Built exposure map for {len(emap)} tickers.")
    print("NVDA topics:", emap.get("NVDA", []))
    print("LLY topics: ", emap.get("LLY", []))
