# Two-stage build: the CPU-only torch wheel is ~200 MB versus ~2.5 GB for the
# CUDA build, and a clinical demo API does not need a GPU. Swap the index-url
# below for GPU inference.
FROM python:3.11-slim AS builder

WORKDIR /build

# opencv needs libGL at runtime; build tools are only needed here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


FROM python:3.11-slim

LABEL org.opencontainers.image.title="alzheimers-detection" \
      org.opencontainers.image.description="VGG19 MRI stage classifier with Grad-CAM"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Checkpoints are mounted, not baked: a 550 MB layer that changes every
# retrain defeats image caching entirely.
VOLUME ["/app/artifacts", "/app/data"]

# Run unprivileged.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "alzheimers.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
