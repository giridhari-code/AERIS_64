# Multi-stage Dockerfile for NeuroField (inference). Put a safetensors checkpoint FOLDER at /app/checkpoints/model
# Non-root, minimal attack surface, CUDA optional.

ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121

# ---------- builder ----------
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY configs/ ./configs/

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url ${TORCH_INDEX} && \
    pip install --no-cache-dir ".[serve]"

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Security: non-root user
RUN groupadd -r neurofield && useradd -r -g neurofield -u 10001 neurofield

WORKDIR /app

# Copy only what is needed
# whole /usr/local (not a hardcoded python3.11 path, which broke when PYTHON_VERSION changed)
COPY --from=builder /usr/local /usr/local
COPY configs/ /app/configs/
COPY deploy/docker/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh && \
    mkdir -p /app/checkpoints /app/logs /app/data && \
    chown -R neurofield:neurofield /app

USER neurofield

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NEUROFIELD_CONFIG=/app/configs/default.yaml \
    NEUROFIELD_CHECKPOINT=/app/checkpoints/model \
    NEUROFIELD_DEVICE=cuda \
    NEUROFIELD_HOST=0.0.0.0 \
    NEUROFIELD_PORT=8000 \
    NEUROFIELD_LOG_JSON=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
