#!/bin/bash
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

unset SINGULARITY_BIND
unset APPTAINER_BIND
unset SINGULARITY_BINDPATH
unset APPTAINER_BINDPATH
unset TMPDIR
unset TEMP
unset TMP

cd "${SLURM_SUBMIT_DIR}"
[ ! -f .env ] && cd ..
source agentic/bin/activate

bash run_puhti.sh

echo "Waiting for services to initialise..."
sleep 120

echo ""
echo "========================================"
echo "UI READY: http://$(hostname):7860"
echo "Connect from your desktop:"
echo "  ssh -L 7860:$(hostname):7860 $(hostname)"
echo "========================================"

sleep infinity
