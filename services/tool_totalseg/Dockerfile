# =============================================================================
# services/tool_totalseg/Dockerfile
#
# Docker image for the TotalSegmentator tool service.
# Functionally equivalent to tool_totalseg.def (Apptainer).
#
# Build (from repo root):
#   docker build -f services/tool_totalseg/Dockerfile -t agseg-totalseg .
#
# Run:
#   docker run -p 8011:8011 agseg-totalseg
# =============================================================================

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    pydantic==2.8.2 \
    numpy==2.2.6 \
    pillow==12.1.0 \
    nibabel==5.3.3 \
    totalsegmentator

COPY services/tool_totalseg/api.py /app/api.py

RUN mkdir -p /input /output

EXPOSE 8011

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8011"]
