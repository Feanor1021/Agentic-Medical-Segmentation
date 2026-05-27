
# =============================================================================
# services/tool_voxtell/Dockerfile
#
# Docker image for the VoxTell tool service.
# Functionally equivalent to tool_voxtell.def (Apptainer).
#
# Build (from repo root):
#   docker build -f services/tool_voxtell/Dockerfile -t agseg-voxtell .
#
# Run:
#   docker run --gpus all -p 8012:8012 \
#     -e VOXTELL_MODEL_DIR=/app/voxtell_weights/voxtell_v1.1 \
#     agseg-voxtell
# =============================================================================

FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV VOXTELL_MODEL_DIR=/app/voxtell_weights/voxtell_v1.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    pydantic==2.8.2 \
    requests \
    huggingface_hub \
    python-multipart

# Install VoxTell from source
RUN git clone https://github.com/MIC-DKFZ/VoxTell.git /tmp/voxtell && \
    cd /tmp/voxtell && pip install --no-cache-dir . && \
    rm -rf /tmp/voxtell

# Download model weights
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='mrokuss/VoxTell', \
    allow_patterns=['voxtell_v1.1/*', '*.json'], \
    local_dir='/app/voxtell_weights')"

# Download text encoder
RUN mkdir -p /app/hf_cache && \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Qwen/Qwen3-Embedding-4B', \
    local_dir='/app/hf_cache')"

COPY services/tool_voxtell/api.py /app/api.py

RUN mkdir -p /data/input /data/output

EXPOSE 8012

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8012"]
