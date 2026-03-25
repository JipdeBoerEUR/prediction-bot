import os
import re
from transformers import pipeline
import torch
import warnings

# Suppress warnings from transformers or edgartools
warnings.filterwarnings("ignore")

# Initialize FinBERT for text analysis (shared with sentiment_ai to save memory if needed)
print("[SEC-AI] Loading FinBERT model for regulatory parsing (ProsusAI/finbert)...")
try:
    sec_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
except Exception as e:
    print(f"[SEC-AI] Pipeline init failed (might be loaded elsewhere): {e}")

def chunk_text(text: str, max_words: int = 50) -> list:
    """Break a massive SEC section into smaller paragraph chunks for BERT to digest."""
    # Split by double newline or simple sentence split if it's one block
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 20]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for p in paragraphs:
        word_count = len(p.split())
        if current_length + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [p]
            current_length = word_count
        else:
            current_chunk.append(p)
            current_length += word_count
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def analyze_sec_filings(ticker: str, max_chunks: int = 15) -> dict:
    """
    Downloads the most recent 10-K and 10-Q for the company and analyzes
    the Management's Discussion & Analysis (MD&A) and Risk Factors (Item 1A).
    """
    try:
        from edgar import set_identity, Company
        
        # edgartools requires an identity set to download from the SEC EDGAR API legally
        set_identity(f"PredictionBotAI (admin@{ticker.lower()}.com)")
        
        company = Company(ticker)
        if not company:
            return {"is_passed": True, "score": 0.0, "reason": "Ticker not found in EDGAR database."}
            
        print(f"[SEC-AI] Digesting latest 10-K and 10-Q for {ticker}...")
        
        # Grab the latest 10-K
        filings = company.get_filings(form="10-K")
        if len(filings) == 0:
            return {"is_passed": True, "score": 0.0, "reason": "No 10-K filings found."}
            
        latest_filing = filings[0]
        tenk = latest_filing.obj()
        
        if not tenk:
             return {"is_passed": True, "score": 0.0, "reason": "Could not parse 10-K document object."}
             
        # Extract the key sections. (Note: edgartools makes accessing Items like a dict)
        mda_text = ""
        risk_text = ""
        
        try:
             # Item 7 is Management's Discussion and Analysis of Financial Condition
             mda_text = tenk['Item 7']
        except Exception:
             pass
             
        try:
             # Item 1A is Risk Factors
             risk_text = tenk['Item 1A']
        except Exception:
             pass
             
        if not mda_text and not risk_text:
             # Sometimes the parser fails on older unstructured HTML forms.
             return {"is_passed": True, "score": 0.0, "reason": "Could not extract MD&A or Risk Factors."}
             
        # We only want to analyze a sample of the text because 10-Ks are 100+ pages long.
        # We'll take the first few chunks of MD&A (usually the executive summary of the business climate)
        # And the first few chunks of Risk Factors (the most heavily weighted risks).
        
        text_to_analyze = ""
        if mda_text:
            text_to_analyze += mda_text[:5000] # First 5000 characters of MD&A
            
        if risk_text:
            text_to_analyze += " " + risk_text[:3000] # First 3000 characters of Risk
            
        chunks = chunk_text(text_to_analyze, max_words=60)
        chunks = chunks[:max_chunks] # Limit to max_chunks (e.g. 15 chunks = ~900 words)
        
        if not chunks:
            return {"is_passed": True, "score": 0.0, "reason": "Extracted text was empty."}
            
        # Run FinBERT on the extracted chunks
        results = sec_pipe(chunks)
        
        label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        scores = []
        for res in results:
            label = res['label']
            confidence = res['score']
            scores.append(label_map[label] * confidence)
            
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # SEC filings are inherently written in "legalese" which can lean negative.
        # So we use a VERY lenient threshold. If avg score is lower than -0.6, they are explicitly terrified.
        is_passed = avg_score >= -0.6
        
        reason = f"10-K Corporate SEC Sentiment Score: {avg_score:.2f} (Analyzed {len(chunks)} key sections)"
        return {
            "is_passed": is_passed,
            "score": avg_score,
            "reason": reason,
            "full_text": text_to_analyze # Pass the sampled text (chunks of MD&A/Risk) for RAG
        }

    except ImportError:
        return {"is_passed": True, "score": 0.0, "reason": "Missing edgartools library."}
    except Exception as e:
        print(f"[SEC-AI] Error analyzing SEC for {ticker}: {e}")
        return {"is_passed": True, "score": 0.0, "reason": f"SEC Parsing Error: {e}"}

if __name__ == "__main__":
    # Smoke test on a known ticker
    res = analyze_sec_filings("AAPL")
    print(res)
