# =============================================================================
# services/tool_totalseg/tool_totalseg.Dockerfile
#
# Image for the TotalSegmentator tool service.
# TotalSegmentator is installed via pip inside the container (CPU-capable,
# but GPU strongly recommended for reasonable inference speed).
#
# Build
# -----
#   docker build -f services/tool_totalseg/tool_totalseg.Dockerfile -t agseg-totalseg .
#
# Run
# ---
#   docker run -p 8011:8011 agseg-totalseg
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

COPY services/tool_totalseg/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY services/tool_totalseg/api.py /app/api.py

EXPOSE 8011

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8011"]
