# =============================================================================
# services/tool_voxtell/tool_voxtell.Dockerfile
#
# Image for the VoxTell tool service.
# VoxTell weights must be present at VOXTELL_MODEL_DIR inside the container
# (default: /app/voxtell_weights/voxtell_v1.1) — mount or copy them in.
#
# Build
# -----
#   docker build -f services/tool_voxtell/tool_voxtell.Dockerfile -t agseg-voxtell .
#
# Run
# ---
#   docker run -p 8012:8012 \
#     -v /path/to/weights:/app/voxtell_weights \
#     agseg-voxtell
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

COPY services/tool_voxtell/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY services/tool_voxtell/api.py /app/api.py

EXPOSE 8012

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8012"]
