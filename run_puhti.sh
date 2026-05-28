#!/bin/bash
# =============================================================================
# run_puhti.sh
#
# Starts all services on Puhti HPC cluster using Apptainer.
# Missing SIF files are automatically downloaded from HuggingFace.
#
# Usage:
#   cp .env.example .env   # do this once, then fill in your values
#   bash run_puhti.sh
#
# Services and ports (read from .env, defaults shown below):
#   VLM           : 8001  (GPU 0)
#   LLM           : 8002  (GPU 1)
#   TotalSeg      : 8011  (GPU 0)
#   VoxTell       : 8012  (GPU 1)
#   BiomedParse   : 8013  (GPU 1)
#   Orchestrator  : 7860
#
# Logs : logs/<service>.log
# PIDs : logs/<service>.pid
# =============================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
ENV_FILE="${ROOT_DIR}/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] .env file not found."
    echo "  Run first: cp .env.example .env"
    echo "  Then fill in your values."
    exit 1
fi
set -a; source "$ENV_FILE"; set +a
# ---------------------------------------------------------------------------
# Required variable check
# ---------------------------------------------------------------------------
for var in HF_SIF_REPO SCRATCH; do
    if [[ -z "${!var}" || "${!var}" == *"XXXXXXX"* || "${!var}" == *"your-"* ]]; then
        echo "[ERROR] '$var' is not configured in .env."
        exit 1
    fi
done
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIF_DIR="${ROOT_DIR}/apptainer/sif"
LOG_DIR="${ROOT_DIR}/logs"
VLM_PORT="${VLM_PORT:-8001}"
LLM_PORT="${LLM_PORT:-8002}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
TOTALSEG_PORT="${TOTALSEG_PORT:-8011}"
VOXTELL_PORT="${VOXTELL_PORT:-8012}"
BIOMEDPARSE_PORT="${BIOMEDPARSE_PORT:-8013}"
export APPTAINER_CACHEDIR="${SCRATCH}/tmp/apptainer_cache"
export APPTAINER_TMPDIR="${SCRATCH}/tmp/apptainer_tmp"
mkdir -p "${SIF_DIR}" "${LOG_DIR}" \
         "${SCRATCH}/tmp" "${SCRATCH}/outputs" \
         "${SCRATCH}/hf_home" "${SCRATCH}/models" \
         "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
# ---------------------------------------------------------------------------
# Download SIF files from HuggingFace if not present
# ---------------------------------------------------------------------------
SIF_FILES=(vlm llm orchestrator totalsegmentator voxtell biomedparse)
echo "[sif] HuggingFace repo: ${HF_SIF_REPO}"
# Detect whether the new `hf` CLI or legacy `huggingface-cli` is available
HF_CMD=""
if command -v hf &>/dev/null; then
    HF_CMD="hf"
elif command -v huggingface-cli &>/dev/null; then
    HF_CMD="huggingface-cli"
else
    echo "[sif] No HF CLI found, installing huggingface_hub..."
    pip install -q huggingface_hub
    if command -v hf &>/dev/null; then
        HF_CMD="hf"
    elif command -v huggingface-cli &>/dev/null; then
        HF_CMD="huggingface-cli"
    else
        echo "[ERROR] Could not install HF CLI. Install manually: pip install huggingface_hub"
        exit 1
    fi
fi
echo "[sif] Using CLI: ${HF_CMD}"
for name in "${SIF_FILES[@]}"; do
    sif="${SIF_DIR}/${name}.sif"
    if [[ -f "$sif" ]]; then
        echo "[sif] skip  ${name}.sif (already exists)"
    else
        echo "[sif] downloading ${name}.sif ..."
        ${HF_CMD} download "${HF_SIF_REPO}" "${name}.sif" \
            --repo-type dataset \
            --local-dir "${SIF_DIR}"
        if [[ ! -f "$sif" ]]; then
            echo "[ERROR] Failed to download ${name}.sif — check HF_SIF_REPO and file name."
            exit 1
        fi
        echo "[sif] ok    ${name}.sif"
    fi
done
# ---------------------------------------------------------------------------
# Kill stale services
# ---------------------------------------------------------------------------
for svc in orchestrator vlm llm totalseg voxtell biomedparse; do
    pid_file="${LOG_DIR}/${svc}.pid"
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        kill "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null
        rm -f "$pid_file"
    fi
done
for port in 8001 8002 8011 8012 8013 7860; do
    pids=$(lsof -ti tcp:${port} 2>/dev/null)
    if [[ -n "$pids" ]]; then
        echo "[cleanup] killing stale process on port ${port}: ${pids}"
        kill -9 ${pids} 2>/dev/null
    fi
done
pkill -f "voxtell_server\|server:app.*8105" 2>/dev/null || true
sleep 2
rm -f "${LOG_DIR}"/*.log "${LOG_DIR}"/*.pid
# ---------------------------------------------------------------------------
# Export service URLs
# ---------------------------------------------------------------------------
export VLM_URL="http://127.0.0.1:${VLM_PORT}"
export LLM_URL="http://127.0.0.1:${LLM_PORT}"
export TOOL_URL_TOTALSEG="http://127.0.0.1:${TOTALSEG_PORT}"
export TOOL_URL_VOXTELL="http://127.0.0.1:${VOXTELL_PORT}"
export TOOL_URL_BIOMEDPARSE="http://127.0.0.1:${BIOMEDPARSE_PORT}"
export OUTPUT_DIR="${SCRATCH}/outputs"
# ---------------------------------------------------------------------------
# Start services
# ---------------------------------------------------------------------------
echo "[start] vlm (GPU 0)"
nohup apptainer run --nv \
    --bind "${SCRATCH}/models:/models" \
    --bind "${SCRATCH}/hf_home:/models/hf" \
    --bind "${SCRATCH}/tmp:/tmp" \
    --env "HF_HOME=/models/hf" \
    --env "TMPDIR=${SCRATCH}/tmp" \
    --env "PORT=${VLM_PORT}" \
    --env "APPTAINERENV_CUDA_VISIBLE_DEVICES=0" \
    "${SIF_DIR}/vlm.sif" \
    > "${LOG_DIR}/vlm.log" 2>&1 &
echo $! > "${LOG_DIR}/vlm.pid"; echo "  pid: $!"
echo "[start] llm (GPU 1)"
nohup apptainer run --nv \
    --bind "${SCRATCH}/models:/models" \
    --bind "${SCRATCH}/hf_home:/models/hf" \
    --bind "${SCRATCH}/tmp:/tmp" \
    --bind "${ROOT_DIR}/services/llm/server.py:/app/server.py" \
    --env "HF_HOME=/models/hf" \
    --env "TMPDIR=${SCRATCH}/tmp" \
    --env "PORT=${LLM_PORT}" \
    "${SIF_DIR}/llm.sif" \
    > "${LOG_DIR}/llm.log" 2>&1 &
echo $! > "${LOG_DIR}/llm.pid"; echo "  pid: $!"
echo "[start] totalseg (GPU 0)"
nohup apptainer exec --nv \
    --bind "${ROOT_DIR}/services/tool_totalseg/api.py:/app/api.py" \
    --bind "${SCRATCH}/hf_home:/root/.totalsegmentator" \
    --bind "/scratch:/scratch" \
    --bind "${SCRATCH}/tmp:/gradio_tmp" \
    --bind "${SCRATCH}/outputs:/tmp/outputs" \
    --env "TMPDIR=${SCRATCH}/tmp" \
    --env "MPLCONFIGDIR=${SCRATCH}/tmp" \
    --env "APPTAINERENV_CUDA_VISIBLE_DEVICES=0" \
    --pwd /app \
    "${SIF_DIR}/totalsegmentator.sif" \
    uvicorn api:app --host 0.0.0.0 --port "${TOTALSEG_PORT}" \
    > "${LOG_DIR}/totalseg.log" 2>&1 &
echo $! > "${LOG_DIR}/totalseg.pid"; echo "  pid: $!"
echo "[start] voxtell (GPU 1)"
nohup apptainer exec --nv \
    --bind "${ROOT_DIR}/services/tool_voxtell/api.py:/app/api.py" \
    --bind "${SCRATCH}/hf_home:/hf_cache" \
    --bind "/scratch:/scratch" \
    --bind "${SCRATCH}/tmp:/gradio_tmp" \
    --bind "${SCRATCH}/outputs:/tmp/outputs" \
    --env "HF_HOME=/hf_cache" \
    --env "VOXTELL_MODEL_DIR=/app/voxtell_weights/voxtell_v1.1" \
    --env "TMPDIR=${SCRATCH}/tmp" \
    --env "APPTAINERENV_CUDA_VISIBLE_DEVICES=1" \
    --pwd /app \
    "${SIF_DIR}/voxtell.sif" \
    uvicorn api:app --host 0.0.0.0 --port "${VOXTELL_PORT}" \
    > "${LOG_DIR}/voxtell.log" 2>&1 &
echo $! > "${LOG_DIR}/voxtell.pid"; echo "  pid: $!"
echo "[start] biomedparse (GPU 1)"
nohup apptainer exec --nv \
    --bind "${ROOT_DIR}/services/tool_biomedparse/api.py:/app/api.py" \
    --bind "${SCRATCH}/hf_home:/app/hf_cache" \
    --bind "/scratch:/scratch" \
    --bind "${SCRATCH}/tmp:/tmp" \
    --bind "${SCRATCH}/outputs:/tmp/outputs" \
    --bind "${SCRATCH}/tmp:/gradio_tmp" \
    --env "PYTHONUNBUFFERED=1" \
    --env "PYTHONPATH=/app" \
    --env "HF_HOME=/app/hf_cache" \
    --env "HUGGINGFACE_HUB_CACHE=/app/hf_cache" \
    --env "TMPDIR=${SCRATCH}/tmp" \
    --env "APPTAINERENV_CUDA_VISIBLE_DEVICES=1" \
    --pwd /app \
    "${SIF_DIR}/biomedparse.sif" \
    uvicorn api:app --host 0.0.0.0 --port "${BIOMEDPARSE_PORT}" \
    > "${LOG_DIR}/biomedparse.log" 2>&1 &
echo $! > "${LOG_DIR}/biomedparse.pid"; echo "  pid: $!"
echo "[start] orchestrator"
nohup apptainer run --nv \
    --bind "${SCRATCH}/outputs:/tmp/outputs" \
    --bind "${SCRATCH}/tmp:/gradio_tmp" \
    --bind "${ROOT_DIR}/orchestrator:/app" \
    --env "VLM_URL=${VLM_URL}" \
    --env "LLM_URL=${LLM_URL}" \
    --env "TOOL_URL_TOTALSEG=${TOOL_URL_TOTALSEG}" \
    --env "TOOL_URL_VOXTELL=${TOOL_URL_VOXTELL}" \
    --env "TOOL_URL_BIOMEDPARSE=${TOOL_URL_BIOMEDPARSE}" \
    --env "OUTPUT_DIR=/tmp/outputs" \
    --env "GRADIO_TEMP_DIR=/gradio_tmp" \
    --env "TMPDIR=/gradio_tmp" \
    "${SIF_DIR}/orchestrator.sif" \
    > "${LOG_DIR}/orchestrator.log" 2>&1 &
echo $! > "${LOG_DIR}/orchestrator.pid"; echo "  pid: $!"
echo ""
echo "========================================"
echo "Services started."
echo "- VLM:          http://127.0.0.1:${VLM_PORT}"
echo "- LLM:          http://127.0.0.1:${LLM_PORT}"
echo "- TotalSeg:     http://127.0.0.1:${TOTALSEG_PORT}"
echo "- VoxTell:      http://127.0.0.1:${VOXTELL_PORT}"
echo "- BiomedParse:  http://127.0.0.1:${BIOMEDPARSE_PORT}"
echo "- Gradio UI:    http://127.0.0.1:${GRADIO_PORT}"
echo ""
echo "Health check (after 60-90s):"
echo "  for port in 8001 8002 8011 8012 8013; do curl -s http://127.0.0.1:\$port/health; echo; done"
echo ""
echo "Logs: ${LOG_DIR}/"
echo "========================================"
