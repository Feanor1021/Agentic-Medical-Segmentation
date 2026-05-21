# =============================================================================
# services/tool_biomedparse/tool_biomedparse.Dockerfile
#
# Image for the BiomedParse tool service.
# The heavy BiomedParse dependencies (fvcore, iopath, pycocotools, model
# weights) are already baked into the base image — only the FastAPI wrapper
# and its lightweight deps are installed here.
#
# Build
# -----
#   docker build -f services/tool_biomedparse/tool_biomedparse.Dockerfile -t agseg-biomedparse .
#
# Run
# ---
#   docker run --gpus all -p 8013:8013 agseg-biomedparse
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

COPY services/tool_biomedparse/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY services/tool_biomedparse/api.py /app/api.py

EXPOSE 8013

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8013"]
