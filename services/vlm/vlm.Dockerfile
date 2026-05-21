# =============================================================================
# services/vlm/vlm.Dockerfile
#
# GPU image for the VLM service (Hulu-Med).
#
# Build
# -----
#   docker build -f services/vlm/vlm.Dockerfile -t agseg-vlm .
#
# Run
# ---
#   docker run --gpus all -p 8001:8001 \
#     -e HULU_MODEL_ID=ZJU-AI4H/Hulu-Med-7B \
#     -e HF_HOME=/models/hf \
#     -v ./models:/models \
#     agseg-vlm
# =============================================================================

FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/vlm/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY services/vlm/ /app/

ENV HOST=0.0.0.0
ENV PORT=8001
# HuggingFace model cache — mount a volume here to avoid re-downloading
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
