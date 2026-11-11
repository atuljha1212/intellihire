FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only torch keeps the image small
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm

COPY . .
# Pre-download MiniLM + Whisper so the first request is fast. If this fails the image still builds,
# but the first request will download models at runtime (and may time out without network access).
RUN python scripts/download_models.py || echo "WARNING: model pre-download failed; models will be fetched at first use" >&2

# Run as an unprivileged user; /app and the data/models mount points must be writable by it.
RUN useradd --create-home --uid 1000 intellihire \
    && mkdir -p /app/data /app/models \
    && chown -R intellihire:intellihire /app
USER intellihire

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
