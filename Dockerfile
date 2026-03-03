FROM python:3.11-slim

WORKDIR /app

# Install system deps for asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml + stub package so hatchling can resolve deps
COPY pyproject.toml .
RUN mkdir -p engram && echo '__version__ = "0.1.0"' > engram/__init__.py
RUN pip install --no-cache-dir .

# Copy full source (overwrites stub)
COPY . .

EXPOSE 8100

CMD ["uvicorn", "engram.api.app:app", "--host", "0.0.0.0", "--port", "8100"]
