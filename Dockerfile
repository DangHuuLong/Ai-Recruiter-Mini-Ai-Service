# syntax=docker/dockerfile:1
#
# NOT Alpine on purpose: torch (pulled in by sentence-transformers) ships glibc-linked
# manylinux wheels and does not run on musl libc (Alpine) — this must stay Debian-based.
#
# PaddleOCR/PaddlePaddle (requirements-ocr.txt) are intentionally NOT installed — the
# project decided not to use OCR fallback. This is safe: document_text_extraction_
# service.py imports paddleocr inside a try/except ImportError and degrades gracefully
# (logs a warning, returns empty OCR result) when the package is absent, so no code
# change was needed to drop it.
FROM python:3.12-slim
WORKDIR /app

# libgomp1: OpenMP runtime used by torch on CPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install CPU-only torch explicitly first — otherwise pip resolves the default GPU build
# via sentence-transformers' unpinned torch dependency, pulling ~2GB of unused CUDA
# libraries (nvidia-cudnn, cublas, cusolver, triton...) into an image that never runs on
# a GPU. Installing the CPU wheel first satisfies that dependency before pip gets to it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Fine-tuned weights are gitignored (never pushed to GitHub) — this COPY only works
# when models/ already exists in the local build context. A CI runner doing a fresh
# `git clone` + `docker build` will NOT have this directory; the pipeline needs a
# separate step to fetch these first (e.g. from blob storage) before `docker build`.
COPY models ./models

RUN groupadd --system app && useradd --system --gid app --home-dir /home/app --create-home app \
    && chown -R app:app /app
USER app
ENV HOME=/home/app

# Pre-download the shared SBERT base model into the image so the container never depends
# on reaching huggingface.co at runtime. Without this, a cold container with no cached
# copy silently falls back to an UNTRAINED random-weights encoder (sentence-transformers'
# own behavior when it can't resolve the name) instead of erroring loudly — confirmed by
# running this image fresh and seeing "Creating a new one with mean pooling" in the logs.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]