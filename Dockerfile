FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libgomp1 is strictly required by LightGBM/OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and model artifacts
COPY api/ api/
COPY artifacts/ artifacts/
COPY models/ models/
COPY src/ src/

ENV PORT=8000
EXPOSE 8000

# Railway provides $PORT dynamically; default to 8000 if not set
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
