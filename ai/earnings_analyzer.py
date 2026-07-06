import yfinance as yf
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

print("[EARNINGS-AI] Loading FinBERT model for transcripts (ProsusAI/finbert)...")
try:
    earnings_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
except Exception as e:
    print(f"[EARNINGS-AI] Pipeline init failed: {e}")

def get_earnings_articles(ticker: str, max_articles: int = 5) -> list:
    """Fetch recent news articles specifically related to earnings/results."""
    stock = yf.Ticker(ticker)
    news = stock.news
    earnings_texts = []
    
    if not news:
        return []
        
    for item in news:
        title = item.get('title', '').lower()
        # Look for earnings-related keywords
        if any(kw in title for kw in ['earnings', 'results', 'q1', 'q2', 'q3', 'q4', 'quarter', 'guidance', 'revenue']):
            # Usually yfinance gives us a summary or a link. Sometimes the publisher provides a full summary
            summary = item.get('summary', '') or item.get('title', '')
            earnings_texts.append(summary)
            if len(earnings_texts) >= max_articles:
                break
                
    return earnings_texts

def analyze_earnings_sentiment(ticker: str) -> dict:
    """
    Attempts to evaluate management tone by analyzing recent earnings-related text.
    First tries to find earnings news/summaries, and scores them using FinBERT.
    """
    try:
        print(f"[EARNINGS-AI] Searching for recent earnings communication for {ticker}...")
        
        texts = get_earnings_articles(ticker)
        
        if not texts:
            return {"is_passed": True, "score": 0.0, "reason": "No recent earnings-related text found."}
            
        # FinBERT supports 512 tokens. We process each text summary individually.
        label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        scores = []
        
        for text in texts:
            # Chunk the text to prevent token overflow
            words = text.split()
            chunks = [" ".join(words[i:i+50]) for i in range(0, len(words), 50)][:5] # Up to 5 chunks per article
            
            for chunk in chunks:
                if len(chunk) < 10:
                    continue
                res = earnings_pipe(chunk)[0]
                label = res['label']
                confidence = res['score']
                scores.append(label_map[label] * confidence)
                
        if not scores:
            return {"is_passed": True, "score": 0.0, "reason": "Extracted earnings text was unscoreable."}
            
        avg_score = sum(scores) / len(scores)
        
        # Earnings reports are inherently volatile. If the score is deeply negative, 
        # it means guidance was slashed or management explicitly cited material weakness.
        # -0.4 threshold is robust.
        is_passed = avg_score >= -0.4
        
        reason = f"Earnings Management Tone Score: {avg_score:.2f} (Analyzed {len(scores)} text chunks)"
        
        return {
            "is_passed": is_passed,
            "score": avg_score,
            "reason": reason
        }
        
    except Exception as e:
        print(f"[EARNINGS-AI] Error parsing earnings for {ticker}: {e}")
        return {"is_passed": False, "score": 0.0, "reason": f"AI Parsing Error (fail-safe — trade blocked): {e}"}

if __name__ == "__main__":
    res = analyze_earnings_sentiment("AAPL")
    print("\nResult:", res)
