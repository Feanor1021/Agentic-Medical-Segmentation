#!/bin/bash
# =============================================================================
# ui_job.sh
#
# SLURM batch script that launches the full Agentic Segmentation UI on Puhti.
# Requests 3× V100 GPUs, starts all microservices via run_puhti.sh, waits for
# them to become ready, then prints an SSH tunnel command so the user can
# open the Gradio interface from their local browser.
#
# Usage:
#   sbatch ui_job.sh
#
# After the job starts, check the log for the tunnel command:
#   tail -f logs/ui_job_<JOBID>.out
# =============================================================================

#SBATCH --job-name=agentic_ui
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:3
#SBATCH --time=4:00:00
#SBATCH --output=logs/ui_job_%j.out
#SBATCH --error=logs/ui_job_%j.err

# ---------------------------------------------------------------------------
# Clean environment inherited from interactive sessions
# ---------------------------------------------------------------------------
unset SINGULARITY_BIND
unset APPTAINER_BIND
unset SINGULARITY_BINDPATH
unset APPTAINER_BINDPATH
unset TMPDIR
unset TEMP
unset TMP

# ---------------------------------------------------------------------------
# Start all services (VLM, LLM, tools, orchestrator)
# ---------------------------------------------------------------------------
cd /scratch/project_2016517/furkan/agentic-seg
source agentic/bin/activate
bash run_puhti.sh

# ---------------------------------------------------------------------------
# Wait for services to be ready (typically 60-90s for model loading)
# ---------------------------------------------------------------------------
echo "Waiting for services to initialise..."
sleep 120

# ---------------------------------------------------------------------------
# Print connection instructions
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "UI READY: http://$(hostname):7860"
echo "Connect from your desktop:"
echo "  ssh -L 7860:$(hostname):7860 $(hostname)"
echo "========================================"

# Keep the job alive so the UI stays accessible
sleep infinity
