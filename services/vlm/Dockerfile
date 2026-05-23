FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# ---- python deps (VLM only) ----
COPY services/vlm/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# ---- app ----
COPY services/vlm /app
ENV HOST=0.0.0.0
ENV PORT=8000
# HuggingFace model cache — mount a volume here to persist across runs
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
EXPOSE 8000
CMD ["python", "api.py"]
