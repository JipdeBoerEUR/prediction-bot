import chromadb
from sentence_transformers import SentenceTransformer
from google import genai as google_genai
import re
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore")

# --- SENTENCE TRANSFORMER MODEL (Loaded once globally for efficiency) ---
_embedding_model: Optional[SentenceTransformer] = None

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print("[RAG-DB] Loading local embedding model: all-MiniLM-L6-v2...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def _embed(texts: List[str]) -> List[List[float]]:
    """Encode a list of strings into vectors using local sentence-transformers."""
    model = _get_embedding_model()
    return model.encode(texts).tolist()

# --- GEMINI SUMMARIZER (FREE TIER via google-genai) ---
class GeminiSummarizer:
    def __init__(self, api_key: str):
        self.client = google_genai.Client(api_key=api_key)

    def summarize_risks(self, ticker: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "No specific risk factors found in the context provided."
            
        combined_context = "\n\n".join(context_chunks)
        prompt = f"""You are a senior financial risk auditor. I am providing text snippets from the latest 10-K filing for {ticker}.
Summarize the most critical risks focused on "Debt", "Lawsuits", "Bankruptcy", or "Material Weakness".

Context from 10-K:
{combined_context}

Provide a concise, 2-3 sentence executive summary. If no major red flags found, state the filings appear standard/stable. Be direct."""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating RAG summary: {e}"

# --- RAG LIBRARY MANAGER ---
class RagLibrary:
    def __init__(self, db_path: str = "./chroma_db", gemini_api_key: Optional[str] = None):
        # Use PersistentClient to store data across bot runs
        self.client = chromadb.PersistentClient(path=db_path)
        self.summarizer = GeminiSummarizer(gemini_api_key) if gemini_api_key else None
        
    def _collection_exists(self, ticker: str) -> bool:
        try:
            self.client.get_collection(name=f"filing_{ticker.lower()}")
            return True
        except Exception:
            return False
        
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks by paragraph, with sentence-level fallback."""
        # Try paragraph splitting first (double newline)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 50]
        
        # Fallback for dense blocks with no paragraph breaks
        if len(paragraphs) < 3:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            grouped: List[str] = []
            for i in range(0, len(sentences), 3):
                chunk = " ".join(sentences[i:i+3]).strip()
                if len(chunk) > 50:
                    grouped.append(chunk)
            if grouped:
                paragraphs = grouped
                
        return paragraphs[:100]  # Cap at 100 chunks
        
    def add_filing(self, ticker: str, text: str):
        """Chunks, embeds, and upserts a document using raw numpy embeddings."""
        if not text or len(text.strip()) < 200:
            print(f"[RAG-DB] Skipping {ticker}: extracted 10-K text is too short or empty.")
            return
            
        print(f"[RAG-DB] Processing 10-K for {ticker} into Vector DB...")
        
        # Get or create a plain collection (no custom embedding function stored in Chroma)
        collection = self.client.get_or_create_collection(name=f"filing_{ticker.lower()}")
        
        chunks = self._chunk_text(text)
        if not chunks:
            print(f"[RAG-DB] No usable paragraphs extracted for {ticker}.")
            return
        
        # Compute embeddings locally and pass them directly — avoids ChromaDB EF API issues
        embeddings = _embed(chunks)
        ids = [f"{ticker}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"ticker": ticker, "source": "10-K"} for _ in chunks]
        
        # upsert prevents duplicate ID crashes on repeated runs
        collection.upsert(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
        print(f"[RAG-DB] Successfully embedded {len(chunks)} chunks for {ticker}.")

    def query_risks(self, ticker: str) -> str:
        """Searches for risk-related chunks and generates a Gemini summary."""
        try:
            collection = self.client.get_collection(name=f"filing_{ticker.lower()}")
        except Exception:
            return "No stored filings found for this ticker in Vector DB."
        
        count = collection.count()
        n = min(5, count)
        if n == 0:
            return "Vector DB collection exists but contains no documents."
        
        # Embed the query locally and pass directly
        query_embedding = _embed(["debt obligations, lawsuits, legal proceedings, bankruptcy risk, material weakness"])
        results = collection.query(query_embeddings=query_embedding, n_results=n)
        
        chunks: List[str] = results['documents'][0]
        
        if self.summarizer:
            return self.summarizer.summarize_risks(ticker, chunks)
        else:
            return "Summarizer not configured. Relevant chunks found but no narrative generated."

# --- INTEGRATED WRAPPER FOR MAIN.PY ---
def generate_rag_risk_report(ticker: str, ten_k_text: Optional[str] = None) -> str:
    """
    Main entry point for the bot pipeline.
    Fails open — returns a safe message string on any error, never crashes the pipeline.
    """
    GEMINI_KEY = "AIzaSyAqrZt85JOm_2KdN_AQg_KqNN5nfMQJbKs"
    
    try:
        library = RagLibrary(gemini_api_key=GEMINI_KEY)
        
        if not library._collection_exists(ticker):
            if ten_k_text and len(ten_k_text.strip()) >= 200:
                library.add_filing(ticker, ten_k_text)
            else:
                return "10-K text not available for RAG processing (EDGAR data may be unavailable)."
                
        return library.query_risks(ticker)
        
    except Exception as e:
        print(f"[RAG-DB] Non-fatal error in RAG pipeline for {ticker}: {e}")
        return f"RAG Risk Analysis unavailable: {e}"

if __name__ == "__main__":
    sample_text = """
    The company faces significant debt obligations totaling $45 billion, 
    due to mature over the next 3-5 years in a rising interest rate environment.

    Legal Proceedings: The company is the defendant in several class-action lawsuits 
    alleging patent infringement and consumer privacy violations that could 
    result in material damages exceeding $2 billion.

    Management warns of potential liquidity constraints. The board has identified 
    material weaknesses in internal controls over financial reporting. 
    Refinancing risk remains elevated. The company may be unable to service debt.
    """
    result = generate_rag_risk_report("TESTCO3", sample_text)
    print("\n=== RAG RISK REPORT ===")
    print(result)
