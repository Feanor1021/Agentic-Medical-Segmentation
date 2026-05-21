# =============================================================================
# services/llm/llm.Dockerfile
#
# GPU image for the LLM service (Qwen).
# Based on the official PyTorch CUDA runtime image.
#
# Build
# -----
#   docker build -f services/llm/llm.Dockerfile -t agseg-llm .
#
# Run
# ---
#   docker run --gpus all -p 8002:8002 \
#     -e QWEN_MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
#     -e HF_HOME=/models/hf \
#     -v ./models:/models \
#     agseg-llm
# =============================================================================

FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/llm/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY services/llm/server.py /app/server.py

ENV PYTHONUNBUFFERED=1
# HuggingFace model cache — mount a volume here to avoid re-downloading
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf

EXPOSE 8002

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
