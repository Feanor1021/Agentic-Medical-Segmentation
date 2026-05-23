#!/bin/bash
# =============================================================================
# run_all.sh
#
# Starts services directly with Python (without Apptainer).
# Use for development/debug only; use run_puhti.sh for production.
#
# Services started:
#   VLM          : services/vlm/api.py        → port 8001
#   LLM          : services/llm/server.py     → port 8002
#   Orchestrator : orchestrator/app.py        → port 7860
#
# Logs: logs/vlm.log, logs/llm.log, logs/orchestrator.log
# =============================================================================

cd /scratch/project_2016517/furkan/agentic-seg
source agentic/bin/activate

# Kill stale processes
pkill -f "api.py" 2>/dev/null
pkill -f "server:app" 2>/dev/null
pkill -f "app.py" 2>/dev/null
sleep 3

echo "[start] VLM"
PORT=8001 HF_HOME=/scratch/project_2016517/furkan/hf_home \
TRANSFORMERS_CACHE=/scratch/project_2016517/furkan/hf_home \
HULU_MODEL_ID=ZJU-AI4H/Hulu-Med-4B \
python services/vlm/api.py > logs/vlm.log 2>&1 &
echo "VLM pid: $!"

echo "[start] LLM"
HF_HOME=/scratch/project_2016517/furkan/hf_home \
python -m uvicorn server:app --host 0.0.0.0 --port 8002 \
    --app-dir services/llm > logs/llm.log 2>&1 &
echo "LLM pid: $!"

echo "[start] Orchestrator"
export VLM_URL=http://127.0.0.1:8001
export LLM_URL=http://127.0.0.1:8002
python orchestrator/app.py > logs/orchestrator.log 2>&1 &
echo "Orchestrator pid: $!"