FROM python:3.10-slim

WORKDIR /app

COPY app /app/app

RUN pip install --no-cache-dir fastapi uvicorn sentence-transformers qdrant-client

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]