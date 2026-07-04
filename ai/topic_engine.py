# ai/topic_engine.py
# BERTopic-powered Topical Trading Engine.
#
# Discovers emerging news narratives from a headline corpus,
# measures topic momentum (velocity of mentions), scores sentiment
# on each topic cluster, and maps topics to exposed tickers.
#
# References:
#   - BERTopic (Grootendorst, 2022) — https://maartengr.github.io/BERTopic
#   - "Topic-Sentiment Dual-Factor Models" (arXiv 2024)
#   - "News-Driven Momentum" (SSRN/RavenPack 2024)

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class DiscoveredTopic:
    """A topic cluster found by BERTopic in the current headline corpus."""
    topic_id:       int
    label:          str              # Human-readable label (top 4 words joined)
    top_words:      List[str]        # Top-10 representative words from BERTopic
    headline_count: int              # Number of headlines in this cluster
    momentum:       float            # Velocity score [0, 1] vs. prior period
    sentiment:      float            # FinBERT avg sentiment [-1, +1]
    headlines:      List[str] = field(default_factory=list)  # Raw headlines


@dataclass
class TopicCandidate:
    """A (ticker, topic) pair with all computed signals."""
    ticker:         str
    topic_label:    str
    topic_id:       int
    topic_momentum: float
    topic_sentiment: float
    exposure_score: float
    headline_count: int
    headlines:      List[str]        # Topic headlines (for downstream FinBERT re-scoring)
    composite_score: float = 0.0    # Combined momentum × sentiment × exposure


# ── Topic Engine ─────────────────────────────────────────────────────────────

class TopicEngine:
    """
    Runs BERTopic over a rolling news headline corpus.

    Usage:
        engine = TopicEngine()
        headlines = aggregate_headlines()   # from news_aggregator
        topics    = engine.discover_topics(headlines)
        candidates = engine.score_topic_candidates(topics, tickers)
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        max_topics: int = 8,
        lookback_hours: int = 6,
        prior_headlines: Optional[List[str]] = None,
    ):
        """
        Args:
            min_cluster_size: Minimum headlines to form a BERTopic cluster.
            max_topics:       Keep only the top-N topics by headline count.
            lookback_hours:   For momentum calc — prior window size.
            prior_headlines:  Headlines from the previous period (for velocity).
        """
        self.min_cluster_size = min_cluster_size
        self.max_topics = max_topics
        self.lookback_hours = lookback_hours
        self.prior_headlines = prior_headlines or []
        self._bertopic_model = None   # lazy-loaded

    # ── BERTopic lazy loader ──────────────────────────────────────────────────

    def _get_bertopic(self):
        """Lazy-load BERTopic with lightweight settings suitable for CPU."""
        if self._bertopic_model is not None:
            return self._bertopic_model

        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from umap import UMAP
            from hdbscan import HDBSCAN

            embedding_model = SentenceTransformer(
                "all-MiniLM-L6-v2",   # 80MB, fast CPU inference
                device="cpu",
            )
            umap_model = UMAP(
                n_neighbors=5,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
                low_memory=True,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=1,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            self._bertopic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                top_n_words=10,
                nr_topics="auto",
                calculate_probabilities=False,
                verbose=False,
            )
            logger.info("[topic_engine] BERTopic model initialised (CPU mode, all-MiniLM-L6-v2).")
        except ImportError as e:
            logger.error("[topic_engine] BERTopic not installed: %s. Run: pip install bertopic sentence-transformers umap-learn hdbscan", e)
            raise

        return self._bertopic_model

    # ── Sentiment scoring ─────────────────────────────────────────────────────

    def _score_sentiment_batch(self, texts: List[str]) -> float:
        """
        Run FinBERT sentiment on a batch of texts, return mean score.
        Falls back to VADER if FinBERT unavailable.
        """
        if not texts:
            return 0.0
        try:
            from transformers import pipeline
            # Reuse a module-level pipeline if already loaded
            global _finbert_pipeline
            if "_finbert_pipeline" not in globals() or _finbert_pipeline is None:
                _finbert_pipeline = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=-1,   # CPU
                    top_k=1,
                )

            label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            scores = []
            for text in texts[:20]:   # cap at 20 to control latency
                try:
                    result = _finbert_pipeline(text[:512], truncation=True)
                    label  = result[0][0]["label"].lower()
                    score  = result[0][0]["score"] * label_map.get(label, 0.0)
                    scores.append(score)
                except Exception:
                    pass
            return float(np.mean(scores)) if scores else 0.0

        except Exception:
            # VADER fallback
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                scores = [analyzer.polarity_scores(t)["compound"] for t in texts[:20]]
                return float(np.mean(scores)) if scores else 0.0
            except Exception:
                return 0.0

    # ── Momentum calculation ──────────────────────────────────────────────────

    def _compute_momentum(
        self,
        topic_words: List[str],
        current_count: int,
    ) -> float:
        """
        Compute topic velocity:
          momentum = (current_count - prior_count) / max(prior_count, 1)
          Normalised to [0, 1] using sigmoid-style compression.

        Args:
            topic_words:   Representative words for this topic.
            current_count: Headlines in this topic cluster right now.

        Returns:
            Momentum score in [0.0, 1.0].
        """
        if not self.prior_headlines:
            # No prior period data — treat all topics as equal momentum
            return min(current_count / 20.0, 1.0)

        pattern = re.compile(
            "|".join(re.escape(w) for w in topic_words[:5]),
            re.IGNORECASE,
        )
        prior_count = sum(
            1 for h in self.prior_headlines
            if pattern.search(h)
        )

        velocity = (current_count - prior_count) / max(prior_count, 1)
        # Sigmoid-compress: velocity=0 → 0.5; velocity=1 → 0.73; velocity=-1 → 0.27
        # We subtract 0.5 so no change = 0.0 base and positive change is positive.
        momentum = (1.0 / (1.0 + np.exp(-velocity))) - 0.5
        # Map to [0, 1]: momentum in [-0.5, 0.5] → [0, 1]
        return float(max(0.0, min(1.0, momentum + 0.5)))

    # ── Fallback topic discovery (LDA/keyword-based) ──────────────────────────

    def _fallback_topic_discovery(self, titles: List[str]) -> List[DiscoveredTopic]:
        """
        Simple keyword-cluster fallback when BERTopic can't run (too few headlines,
        import error, etc.). Groups headlines by common financial theme keywords.
        """
        theme_seeds = {
            "ai_chips":        ["nvidia", "gpu", "ai chip", "h100", "blackwell", "semiconductor"],
            "interest_rates":  ["fed", "federal reserve", "rate hike", "rate cut", "fomc", "inflation", "yields"],
            "earnings":        ["earnings", "revenue", "eps", "beat", "miss", "quarterly results", "profit"],
            "merger_ma":       ["merger", "acquisition", "buyout", "deal", "takeover", "acquisition"],
            "tech_layoffs":    ["layoff", "job cuts", "workforce reduction", "restructuring", "headcount"],
            "biotech_fda":     ["fda", "drug approval", "clinical trial", "phase 3", "breakthrough therapy"],
            "ev_auto":         ["electric vehicle", "ev", "tesla", "autonomous", "battery"],
            "energy":          ["oil", "opec", "crude", "natural gas", "energy prices", "refinery"],
            "crypto":          ["bitcoin", "crypto", "ethereum", "blockchain", "defi", "stablecoin"],
            "trade_tariffs":   ["tariff", "trade war", "china", "supply chain", "export", "import"],
        }

        topics: List[DiscoveredTopic] = []
        used: set = set()

        for theme_key, seeds in theme_seeds.items():
            pattern = re.compile("|".join(re.escape(s) for s in seeds), re.IGNORECASE)
            matched = [t for i, t in enumerate(titles) if pattern.search(t) and i not in used]
            if len(matched) < self.min_cluster_size:
                continue

            for i, t in enumerate(titles):
                if pattern.search(t):
                    used.add(i)

            label = theme_key.replace("_", " ").title()
            topics.append(DiscoveredTopic(
                topic_id=len(topics),
                label=label,
                top_words=seeds[:8],
                headline_count=len(matched),
                momentum=0.5,   # will be refined below
                sentiment=0.0,  # will be refined below
                headlines=matched[:30],
            ))

        return topics[:self.max_topics]

    # ── Primary: BERTopic discovery ───────────────────────────────────────────

    def discover_topics(
        self,
        headlines: List[Dict],
        use_bertopic: bool = True,
    ) -> List[DiscoveredTopic]:
        """
        Run topic discovery on a list of headline dicts
        (each dict must have at least a 'title' key).

        Args:
            headlines:   Output of news_aggregator.aggregate_headlines().
            use_bertopic: Use BERTopic if True, keyword fallback if False.

        Returns:
            List of DiscoveredTopic sorted by momentum desc.
        """
        titles = [h["title"] for h in headlines if h.get("title")]

        if len(titles) < self.min_cluster_size:
            logger.warning("[topic_engine] Only %d headlines — too few for topic discovery.", len(titles))
            return []

        logger.info("[topic_engine] Discovering topics in %d headlines…", len(titles))
        t0 = time.time()

        topics: List[DiscoveredTopic] = []

        if use_bertopic and len(titles) >= 10:
            try:
                model = self._get_bertopic()
                model.fit(titles)
                topic_info = model.get_topic_info()

                for _, row in topic_info.iterrows():
                    tid = int(row["Topic"])
                    if tid == -1:
                        continue   # outliers cluster

                    words_scores = model.get_topic(tid) or []
                    top_words = [w for w, _ in words_scores[:10]]
                    label = "_".join(top_words[:4]).replace("_", " ")

                    # Gather headlines assigned to this topic
                    topic_titles = [
                        titles[i]
                        for i, t in enumerate(model.topics_)
                        if t == tid
                    ][:30]

                    topics.append(DiscoveredTopic(
                        topic_id=tid,
                        label=label,
                        top_words=top_words,
                        headline_count=len(topic_titles),
                        momentum=0.0,   # computed below
                        sentiment=0.0,  # computed below
                        headlines=topic_titles,
                    ))

                logger.info("[topic_engine] BERTopic found %d topics in %.1fs.", len(topics), time.time() - t0)

            except Exception as e:
                logger.warning("[topic_engine] BERTopic failed (%s). Using fallback.", e)
                topics = self._fallback_topic_discovery(titles)
        else:
            topics = self._fallback_topic_discovery(titles)

        # Sort by headline count, take top-N
        topics.sort(key=lambda t: t.headline_count, reverse=True)
        topics = topics[:self.max_topics]

        # Compute momentum + sentiment in parallel (blocking, but fast)
        for topic in topics:
            topic.momentum  = self._compute_momentum(topic.top_words, topic.headline_count)
            topic.sentiment = self._score_sentiment_batch(topic.headlines)
            logger.info(
                "[topic_engine] Topic '%s': %d headlines | momentum=%.2f | sentiment=%.2f",
                topic.label, topic.headline_count, topic.momentum, topic.sentiment,
            )

        # Final sort: momentum first, then sentiment
        topics.sort(key=lambda t: (t.momentum, t.sentiment), reverse=True)
        return topics

    # ── Ticker mapping ────────────────────────────────────────────────────────

    def score_topic_candidates(
        self,
        topics: List[DiscoveredTopic],
        tickers: List[str],
        min_topic_momentum: float = 0.30,
        min_topic_sentiment: float = 0.05,
        min_exposure_score: float = 0.15,
        max_tickers_per_topic: int = 3,
    ) -> List[TopicCandidate]:
        """
        Map discovered topics to the most exposed tickers.

        Args:
            topics:               Output of discover_topics().
            tickers:              Universe of tickers to consider.
            min_topic_momentum:   Skip topics below this momentum threshold.
            min_topic_sentiment:  Skip topics below this sentiment threshold.
            min_exposure_score:   Skip (ticker, topic) pairs below this exposure.
            max_tickers_per_topic: Max tickers per topic to avoid over-concentration.

        Returns:
            List of TopicCandidate sorted by composite_score desc.
        """
        from ai.topic_exposure import build_exposure_map, score_ticker_exposure

        if not topics:
            return []

        exposure_map = build_exposure_map(tickers, use_yfinance=False)
        candidates: List[TopicCandidate] = []

        for topic in topics:
            if topic.momentum < min_topic_momentum:
                logger.debug("[topic_engine] Topic '%s' below momentum threshold (%.2f). Skipping.",
                             topic.label, topic.momentum)
                continue
            if topic.sentiment < min_topic_sentiment:
                logger.debug("[topic_engine] Topic '%s' below sentiment threshold (%.2f). Skipping.",
                             topic.label, topic.sentiment)
                continue

            # Score every ticker for this topic
            ticker_scores: List[Tuple[str, float]] = []
            for ticker in tickers:
                exp = score_ticker_exposure(ticker, topic.label, topic.top_words, exposure_map)
                if exp >= min_exposure_score:
                    ticker_scores.append((ticker, exp))

            # Keep top-N by exposure
            ticker_scores.sort(key=lambda x: x[1], reverse=True)
            ticker_scores = ticker_scores[:max_tickers_per_topic]

            for ticker, exp in ticker_scores:
                composite = (
                    0.4 * topic.momentum
                    + 0.4 * max(0.0, topic.sentiment)
                    + 0.2 * exp
                )
                candidates.append(TopicCandidate(
                    ticker=ticker,
                    topic_label=topic.label,
                    topic_id=topic.topic_id,
                    topic_momentum=topic.momentum,
                    topic_sentiment=topic.sentiment,
                    exposure_score=exp,
                    headline_count=topic.headline_count,
                    headlines=topic.headlines,
                    composite_score=composite,
                ))
                logger.info(
                    "[topic_engine] Candidate %s | topic='%s' | "
                    "momentum=%.2f sentiment=%.2f exposure=%.2f composite=%.2f",
                    ticker, topic.label, topic.momentum, topic.sentiment, exp, composite,
                )

        # Deduplicate: keep only the highest-scoring topic per ticker
        best: Dict[str, TopicCandidate] = {}
        for c in candidates:
            if c.ticker not in best or c.composite_score > best[c.ticker].composite_score:
                best[c.ticker] = c

        result = sorted(best.values(), key=lambda c: c.composite_score, reverse=True)
        logger.info("[topic_engine] %d unique topic candidates after dedup.", len(result))
        return result

    # ── Test helper ───────────────────────────────────────────────────────────

    def test_with_sample_headlines(self) -> None:
        """Quick smoke-test with canned headlines — run from __main__."""
        sample = [
            {"title": "Nvidia announces record AI chip shipments, H100 demand surges"},
            {"title": "NVDA stock hits all-time high as data center revenue doubles"},
            {"title": "AMD unveils MI400 chip to challenge Nvidia in AI training"},
            {"title": "Microsoft Azure AI revenue grows 50% quarter over quarter"},
            {"title": "Federal Reserve signals rate cut in September meeting"},
            {"title": "Inflation data shows CPI cooling, Fed likely to pivot"},
            {"title": "Bond yields fall as Fed officials hint at easing cycle"},
            {"title": "Tesla reports record deliveries, EV demand rebounds strongly"},
            {"title": "Ford F-150 Lightning production increases amid EV market recovery"},
            {"title": "Eli Lilly obesity drug gets FDA approval, shares surge 12%"},
            {"title": "Novo Nordisk GLP-1 supply concerns ease as capacity expands"},
            {"title": "Oil prices fall as OPEC+ increases production targets"},
            {"title": "Exxon Q3 earnings beat on strong refining margins"},
        ]
        topics = self.discover_topics(sample, use_bertopic=False)
        print("\n=== Sample Topic Discovery ===")
        for t in topics:
            print(f"  Topic: '{t.label}' | headlines={t.headline_count} | "
                  f"momentum={t.momentum:.2f} | sentiment={t.sentiment:.2f}")


_finbert_pipeline = None   # module-level cache


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    engine = TopicEngine(min_cluster_size=2)
    engine.test_with_sample_headlines()
