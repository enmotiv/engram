FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8100

CMD ["uvicorn", "engram.api.app:app", "--host", "0.0.0.0", "--port", "8100"]
