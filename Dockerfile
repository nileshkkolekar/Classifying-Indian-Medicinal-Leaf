# syntax=docker/dockerfile:1.7
#
# One image, two entry points: `leaf-api` serves the model over HTTP and
# `leaf-ui` serves the Streamlit front end. A single build means a single ECR
# repository and one layer cache; the ECS task definition picks which role a
# container plays by overriding the command.

# ── Frontend ─────────────────────────────────────────────────────────────
# Built in its own stage so Node never reaches the runtime image: the final
# container ships static files, not a JavaScript toolchain.
FROM node:22-slim AS frontend

WORKDIR /build/frontend

# Dependencies first, so a source-only change reuses the install layer.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ── Builder ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch, installed first so the resolver treats it as satisfied.
# The default Linux wheels bundle CUDA and add roughly 2 GB to an image that
# will never see a GPU on Fargate.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install ".[serve]"

# ── Runtime ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MLC_ENV=production \
    # The package is installed outside a checkout here, so point the config
    # loader at the application directory explicitly.
    MLC_PROJECT_ROOT=/app

# OpenCV links against libGL and glib even when only decoding images.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
 && rm -rf /var/lib/apt/lists/*

# Never run as root: a process that accepts uploads from the internet should
# not be able to write anywhere that matters.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin leaf

COPY --from=builder --chown=leaf:leaf /opt/venv /opt/venv

WORKDIR /app
COPY --chown=leaf:leaf configs/ ./configs/

# The React bundle, served by the API at "/" from the same origin.
COPY --from=frontend --chown=leaf:leaf /build/frontend/dist ./frontend/dist
# Somewhere for a checkpoint to be mounted or downloaded at startup.
RUN mkdir -p /app/artifacts/checkpoints && chown -R leaf:leaf /app/artifacts

USER leaf

EXPOSE 8000 8501

# Reports unhealthy only if the process is down; a missing checkpoint returns
# 200 with status "degraded", which is a configuration problem rather than a
# reason for the orchestrator to keep restarting the task.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Bind all interfaces: the port is published by the container runtime.
ENV MLC_SERVING__HOST=0.0.0.0

CMD ["leaf-api"]
