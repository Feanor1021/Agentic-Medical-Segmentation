# =============================================================================
# Dockerfile  (module-mode variant)
#
# Alternative orchestrator image that runs app.py as a Python module
# (python -m orchestrator.app) instead of directly (python app.py).
# Use this when the project is structured as a package with __init__.py,
# or when relative imports inside orchestrator/ require the package context.
#
# Build
# -----
#   docker build -f Dockerfile -t agseg-orchestrator-mod .
#
# Run
# ---
#   docker run -p 7860:7860 \
#     -e VLM_URL=http://vlm-service:8000 \
#     -e LLM_URL=http://llm-service:8001 \
#     -e TOOL_URL_TOTALSEG=http://totalseg:8011 \
#     -e TOOL_URL_VOXTELL=http://voxtell:8012 \
#     -e TOOL_URL_BIOMEDPARSE=http://biomedparse:8013 \
#     agseg-orchestrator-mod
# =============================================================================

FROM python:3.10-slim

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- env ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# ---- workdir ----
WORKDIR /app

# ---- install python deps ----
# requirements.txt should only contain orchestrator dependencies (no GPU libs)
COPY orchestrator/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- copy orchestrator code ----
COPY orchestrator /app/orchestrator

# ---- outputs dir (runtime artefacts) ----
RUN mkdir -p /app/outputs

# ---- expose UI port ----
EXPOSE 7860

# ---- run ----
# Run as a module so intra-package imports resolve correctly
CMD ["python", "-m", "orchestrator.app"]