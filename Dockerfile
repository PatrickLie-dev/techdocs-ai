# Dockerfile — TechDocs AI
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn==21.2.0

COPY . .

RUN mkdir -p /app/chroma_db /app/documents

ENV FLASK_ENV=production
ENV PORT=5000
ENV PYTHONPATH=/app

EXPOSE 5000

# Health check — pakai /api/health sesuai app kamu
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

CMD ["python", "-m", "gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--timeout", "120", \
     "src.app:app"]