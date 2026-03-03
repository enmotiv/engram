FROM python:3.11-slim

WORKDIR /app

# Install system deps for asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install dependencies only (from pyproject.toml)
# [ml] extra includes sentence-transformers for local embedding (BAAI/bge-m3)
COPY pyproject.toml .
RUN mkdir -p engram && echo '__version__ = "0.1.0"' > engram/__init__.py
RUN pip install --no-cache-dir ".[ml]"

# Copy full source (overwrites stub), then reinstall in editable mode
# so Python resolves engram from /app/engram/ (includes dreamer, plugins, etc.)
COPY . .
RUN pip install --no-cache-dir -e ".[ml]"

# Pre-download the embedding model so first request isn't slow (~2GB)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

EXPOSE 8100

CMD ["uvicorn", "engram.api.app:app", "--host", "0.0.0.0", "--port", "8100"]
