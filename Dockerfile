# syntax=docker/dockerfile:1

# Linux runtime for local OCR verification.
#
# PaddleOCR/PaddlePaddle CPU inference is explicitly disabled on Windows at
# runtime (app/services/paddleocr_runtime_patch.py checks platform.system())
# because it isn't stable there. This image exists so OCR-dependent code
# paths (document_text_extraction_service.py's PaddleOCR fallback) can be
# exercised locally on a Windows dev machine via Docker instead of only in
# the real Linux deployment target.
FROM python:3.12-slim

WORKDIR /app

# Native libs required by opencv-python (a paddleocr dependency) at import
# time, even for headless/text-only OCR usage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ocr.txt ./
# Cache mount persists downloaded wheels across `docker compose build` retries
# — torch/sentence-transformers/paddlepaddle are large and slow to refetch.
# Install CPU-only torch first from PyTorch's own CPU index: sentence-
# transformers otherwise pulls the default PyPI torch build, which drags in
# multiple gigabytes of CUDA/NVIDIA libraries this CPU-only container never
# uses.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && pip install -r requirements-ocr.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
