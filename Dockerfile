FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages (like hnswlib for ChromaDB)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to cache the pip install step
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables for production (User should pass these at runtime or via .env)
ENV PYTHONUNBUFFERED=1

# Run the main pipeline (Note: This will execute the daily scan and terminate, 
# you should set up a cron job inside the container or outside it if you want 24/7 scanning.
# For now, it just runs the main loop once).
CMD ["python", "main.py"]
