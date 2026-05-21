# =============================================================================
# orchestrator.Dockerfile
#
# Minimal production image for the orchestrator service.
# Contains only the UI + pipeline code (app.py, agents, clients, contracts,
# tool_registry) — NO GPU, NO heavy model weights.
#
# Build
# -----
#   docker build -f orchestrator.Dockerfile -t agseg-orchestrator .
#
# Run
# ---
#   docker run -p 7860:7860 \
#     -e VLM_URL=http://vlm-service:8000 \
#     -e LLM_URL=http://llm-service:8001 \
#     -e TOOL_URL_TOTALSEG=http://totalseg:8011 \
#     -e TOOL_URL_VOXTELL=http://voxtell:8012 \
#     -e TOOL_URL_BIOMEDPARSE=http://biomedparse:8013 \
#     agseg-orchestrator
# =============================================================================

FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY orchestrator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy orchestrator source directly into /app so app.py is at /app/app.py
COPY orchestrator/ /app/

RUN mkdir -p /app/outputs

EXPOSE 7860

CMD ["python", "app.py"]
