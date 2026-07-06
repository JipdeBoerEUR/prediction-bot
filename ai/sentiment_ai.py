import yfinance as yf
from transformers import pipeline

# Initialize the FinBERT sentiment analysis pipeline
# This will download the model (~400MB) on the first run
print("[SENTIMENT] Loading FinBERT model (ProsusAI/finbert)...")
sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")

def analyze_sentiment(ticker: str, num_headlines: int = 5) -> dict:
    """
    Fetches news headlines for a ticker and analyzes sentiment using FinBERT.
    Returns a dict with average score and pass/fail status.
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return {"is_passed": True, "score": 0.0, "reason": "No news found for ticker."}

        # yfinance changed its news schema: items are now
        # {"id": ..., "content": {"title": ...}} instead of {"title": ...}.
        # Support both — the old flat access KeyError'd on every item, which
        # the except below turned into a blanket trade rejection.
        headlines = []
        for row in news[:num_headlines]:
            title = (row.get("content") or {}).get("title") or row.get("title") or ""
            if title.strip():
                headlines.append(title.strip())

        if not headlines:
            return {"is_passed": True, "score": 0.0,
                    "reason": "News present but no parseable headlines (schema change?)."}
        results = sentiment_pipe(headlines)
        
        # Map labels to scores: positive=1, neutral=0, negative=-1
        label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        
        scores = []
        for res in results:
            label = res['label']
            score = res['score'] # Confidence score
            scores.append(label_map[label] * score)
            
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Threshold: if average sentiment is significantly negative, fail the trade
        # -0.3 is a conservative threshold (mostly neutral/slightly mixed is OK)
        is_passed = avg_score >= -0.3
        
        reason = f"Average Sentiment Score: {avg_score:.2f} based on {len(headlines)} headlines."
        return {
            "is_passed": is_passed,
            "score": avg_score,
            "reason": reason,
            "headlines": headlines
        }
        
    except Exception as e:
        print(f"[SENTIMENT] Error analyzing sentiment for {ticker}: {e}")
        # Fail-safe: if AI fails, block the trade rather than proceeding blind
        return {"is_passed": False, "score": 0.0, "reason": f"AI Error (fail-safe — trade blocked): {e}"}

if __name__ == "__main__":
    # Smoke test
    print(analyze_sentiment("AAPL"))