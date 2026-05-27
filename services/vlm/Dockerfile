# =============================================================================
# services/vlm/Dockerfile
#
# Docker image for the Hulu-Med VLM service.
# Functionally equivalent to vlm.def (Apptainer).
#
# Build (from repo root):
#   docker build -f services/vlm/Dockerfile -t agseg-vlm .
#
# Run:
#   docker run --gpus all -p 8001:8001 \
#     -v ./models:/models \
#     -e HF_HOME=/models/hf \
#     agseg-vlm
# =============================================================================

FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/vlm/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y torchvision torch || true && \
    pip install --no-cache-dir -r requirements.txt

COPY services/vlm/api.py /app/api.py

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
