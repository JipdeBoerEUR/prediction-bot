FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies needed by ML packages (hdbscan, umap, chromadb, lxml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to cache the pip install layer
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK VADER lexicon at build time (avoids first-run network call)
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# Copy the rest of the application code
COPY . .

# ── Runtime environment ───────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1

# Required at runtime — pass via docker run -e or docker-compose environment:
#   ALPACA_API_KEY      — Alpaca paper/live key
#   ALPACA_SECRET_KEY   — Alpaca paper/live secret
#   GOOGLE_API_KEY      — Gemini API key (used by rag_library.py)
#   TELEGRAM_BOT_TOKEN  — (optional) Telegram alerts
#   TELEGRAM_CHAT_ID    — (optional) Telegram chat ID
#   SENDER_EMAIL        — (optional) SMTP email alerts
#   SENDER_PASSWORD     — (optional) SMTP password
#   RECEIVER_EMAIL      — (optional) alert recipient

# Run the unified trading daemon
CMD ["python", "main.py"]
