#!/bin/bash
# =============================================================================
# run_puhti.sh
#
# Puhti HPC kümesinde tüm servisleri Apptainer ile başlatır.
# SIF dosyaları yoksa önce build eder.
#
# Servisler ve portlar:
#   VLM           : 8001  (GPU 0)
#   LLM           : 8002  (GPU 1)
#   TotalSeg      : 8011  (GPU 0)
#   VoxTell       : 8012  (GPU 1)
#   BiomedParse   : 8013  (GPU 1)
#   Orchestrator  : 7860
#
# Çalıştırma:
#   bash run_puhti.sh
#
# Port override (opsiyonel):
#   VLM_PORT=8001 LLM_PORT=8002 bash run_puhti.sh
#
# Log dosyaları: logs/<servis>.log
# PID dosyaları: logs/<servis>.pid
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF_DIR="${ROOT_DIR}/apptainer/sif"
LOG_DIR="${ROOT_DIR}/logs"
SCRATCH="/scratch/project_2016517/furkan"

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
         "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" \
         /tmp/yardimf/32685786

# PID dosyalarından eski servisleri durdur
for svc in orchestrator vlm llm totalseg voxtell voxtell_server biomedparse; do
    pid_file="${LOG_DIR}/${svc}.pid"
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        kill "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null
        rm -f "$pid_file"
    fi
done

# Port bazlı temizlik — pid dosyası olmayan kalıntı processleri de öldür
for port in 8001 8002 8011 8012 8013 7860 8105; do
    pids=$(lsof -ti tcp:${port} 2>/dev/null)
    if [[ -n "$pids" ]]; then
        echo "[cleanup] killing stale process on port ${port}: ${pids}"
        kill -9 ${pids} 2>/dev/null
    fi
done

pkill -f "voxtell_server\|server:app.*8105" 2>/dev/null || true

sleep 2

# Eski log dosyalarını temizle
rm -f "${LOG_DIR}"/*.log "${LOG_DIR}"/*.pid

# SIF dosyalarını build et (yoksa)
for def_pair in "vlm:services/vlm/vlm.def" "llm:services/llm/llm.def" "orchestrator:orchestrator/orchestrator.def"; do
    name="${def_pair%%:*}"; def_path="${def_pair##*:}"; sif="${SIF_DIR}/${name}.sif"
    if [[ -f "$sif" ]]; then
        echo "[skip] ${name}.sif already exists."
    else
        echo "[build] ${name}.sif ..."
        cd "${ROOT_DIR}"
        APPTAINER_BIND="" APPTAINER_BINDPATH="" SINGULARITY_BIND="" SINGULARITY_BINDPATH="" \
        apptainer build --fakeroot "${sif}" "${ROOT_DIR}/${def_path}"
        echo "[build] OK"
    fi
done

export VLM_URL="http://127.0.0.1:${VLM_PORT}"
export LLM_URL="http://127.0.0.1:${LLM_PORT}"
export TOOL_URL_TOTALSEG="http://127.0.0.1:${TOTALSEG_PORT}"
export TOOL_URL_VOXTELL="http://127.0.0.1:${VOXTELL_PORT}"
export TOOL_URL_BIOMEDPARSE="http://127.0.0.1:${BIOMEDPARSE_PORT}"

echo "[start] vlm (GPU 0)"
nohup apptainer run --nv \
    --bind "${SCRATCH}/models:/models" \
    --bind "${SCRATCH}/hf_home:/models/hf" \
    --bind "${SCRATCH}/tmp:/tmp" \
    --env "HF_HOME=/models/hf" \
    --env "TRANSFORMERS_CACHE=/models/hf" \
    --env "TMPDIR=${SCRATCH}/tmp" \
    --env "PORT=${VLM_PORT}" \
    --env "APPTAINERENV_CUDA_VISIBLE_DEVICES=0" \
    "${SIF_DIR}/vlm.sif" \
    > "${LOG_DIR}/vlm.log" 2>&1 &
echo $! > "${LOG_DIR}/vlm.pid"; echo "  pid: $!"

export CUDA_VISIBLE_DEVICES=""
unset CUDA_VISIBLE_DEVICES
echo "[start] llm (GPU 1)"
nohup apptainer run --nv \
    --bind "${SCRATCH}/models:/models" \
    --bind "${SCRATCH}/hf_home:/models/hf" \
    --bind "${SCRATCH}/tmp:/tmp" \
    --bind "${ROOT_DIR}/services/llm/server.py:/app/server.py" \
    --env "HF_HOME=/models/hf" \
    --env "TRANSFORMERS_CACHE=/models/hf" \
    --env "TMPDIR=${SCRATCH}/tmp" \
    --env "PORT=${LLM_PORT}" \
    "${SIF_DIR}/llm.sif" \
    > "${LOG_DIR}/llm.log" 2>&1 &
echo $! > "${LOG_DIR}/llm.pid"; echo "  pid: $!"

echo "[start] totalseg (GPU 0)"
nohup apptainer exec --nv \
    --bind "${ROOT_DIR}/services/tool_totalseg/api.py:/app/api.py" \
    --bind "${SCRATCH}/hf_home:/users/yardimf/.totalsegmentator" \
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
    --env "TRANSFORMERS_CACHE=/hf_cache" \
    --env "VOXTELL_MODEL_DIR=/app/voxtell_weights/voxtell_v1.1" \
    --env "TMPDIR=${SCRATCH}/tmp" \
    --env "APPTAINERENV_CUDA_VISIBLE_DEVICES=1" \
    --pwd /app \
    "${SIF_DIR}/voxtell.sif" \
    uvicorn api:app --host 0.0.0.0 --port "${VOXTELL_PORT}" \
    > "${LOG_DIR}/voxtell.log" 2>&1 &
echo $! > "${LOG_DIR}/voxtell.pid"; echo "  pid: $!"

echo "[start] biomedparse"
BIOMEDPARSE_SIF="/scratch/project_2016517/balazs/biomedparse/biomedparse.sif"
if [[ ! -f "${BIOMEDPARSE_SIF}" ]]; then
    echo "  [ERROR] BiomedParse SIF not found: ${BIOMEDPARSE_SIF}"
    echo "  Copy with: cp /scratch/project_2016517/haris/biomedparse/biomedparse.sif ${SIF_DIR}/biomedparse.sif"
else
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
        "${BIOMEDPARSE_SIF}" \
        uvicorn api:app --host 0.0.0.0 --port "${BIOMEDPARSE_PORT}" \
        > "${LOG_DIR}/biomedparse.log" 2>&1 &
    echo $! > "${LOG_DIR}/biomedparse.pid"; echo "  pid: $!"
fi

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
echo "Servisler baslatildi."
echo "- VLM:          http://127.0.0.1:${VLM_PORT}"
echo "- LLM:          http://127.0.0.1:${LLM_PORT}"
echo "- TotalSeg:     http://127.0.0.1:${TOTALSEG_PORT}"
echo "- VoxTell:      http://127.0.0.1:${VOXTELL_PORT}"
echo "- BiomedParse:  http://127.0.0.1:${BIOMEDPARSE_PORT}"
echo "- Gradio UI:    http://127.0.0.1:${GRADIO_PORT}"
echo ""
echo "Health check (30s sonra):"
echo "  curl http://127.0.0.1:${TOTALSEG_PORT}/health"
echo "  curl http://127.0.0.1:${VOXTELL_PORT}/health"
echo "  curl http://127.0.0.1:${BIOMEDPARSE_PORT}/health"
echo "  curl http://127.0.0.1:${LLM_PORT}/health"
echo "========================================"
export OUTPUT_DIR=/scratch/project_2016517/furkan/agentic-seg/output