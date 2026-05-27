# =============================================================================
# services/tool_biomedparse/Dockerfile
#
# Docker image for the BiomedParse v2 tool service.
# Functionally equivalent to tool_biomedparse.def (Apptainer).
#
# Build (from repo root):
#   docker build -f services/tool_biomedparse/Dockerfile \
#     --build-arg HF_TOKEN=hf_... -t agseg-biomedparse .
#
# Run:
#   docker run --gpus all -p 8013:8013 agseg-biomedparse
# =============================================================================

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ARG HF_TOKEN

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV PYTHONPATH=/app/BiomedParse:$PYTHONPATH
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone BiomedParse and install dependencies
RUN git clone https://github.com/microsoft/BiomedParse.git /app/BiomedParse && \
    cd /app/BiomedParse && \
    pip install --no-cache-dir -r assets/requirements/requirements.txt && \
    pip install --no-cache-dir azureml-automl-core "pathspec<1.0.0" opencv-python fvcore iopath

# Detectron2 (needs --no-build-isolation to see system PyTorch)
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/facebookresearch/detectron2.git

# FastAPI wrapper dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    pydantic==2.8.2 \
    requests \
    huggingface_hub \
    python-multipart \
    nibabel

# Download gated model weights (requires HF_TOKEN build arg)
RUN mkdir -p /app/BiomedParse/weights /app/hf_cache && \
    python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download( \
    repo_id='microsoft/BiomedParse', \
    filename='biomedparse_v2.ckpt', \
    local_dir='/app/BiomedParse/weights', \
    token='${HF_TOKEN}' \
)"

COPY services/tool_biomedparse/api.py /app/api.py

RUN mkdir -p /data/input /data/output

EXPOSE 8013

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8013"]
