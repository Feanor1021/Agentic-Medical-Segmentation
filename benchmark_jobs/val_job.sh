#!/bin/bash
#SBATCH --job-name=pipeline_val
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:3
#SBATCH --time=12:00:00
#SBATCH --output=logs/val_job_%j.out
#SBATCH --error=logs/val_job_%j.err

cd "${SLURM_SUBMIT_DIR}"
[ ! -f .env ] && cd ..
source agentic/bin/activate
source .env

echo "[1/4] Killing stale processes..."
pkill -u $USER python
pkill -u $USER apptainer

pip install -q numpy nibabel requests scipy

echo "[2/4] Starting services (run_puhti.sh)..."
bash run_puhti.sh

echo "Waiting for services..."
for port in 8001 8002 8011 8012 8013; do
    until curl -s http://127.0.0.1:$port/health | grep -q "ok"; do
        sleep 10; echo "Port $port not ready yet..."
    done
    echo "Port $port ready!"
done
echo "All services ready!"

echo "[3/4] Starting validation..."
DATASET_CSV="${VAL_DATA_DIR}/cases.csv"
OUTPUT_DIR="${VAL_DATA_DIR}/results"

python -u validate_pipeline.py \
    --dataset_csv $DATASET_CSV \
    --output_dir $OUTPUT_DIR 2>&1

echo "[4/4] Done. Summary:"
python -c "
import json, numpy as np, os
path = '$OUTPUT_DIR/results.jsonl'
if not os.path.exists(path):
    print('ERROR: Results file not found!')
    exit()
results = [json.loads(l) for l in open(path)]
all_dice = [r['dice'] for r in results if r['dice'] is not None]
nonzero = [d for d in all_dice if d > 0.01]
print('='*40)
print(f'Total cases  : {len(results)}')
print(f'Mean Dice    : {np.mean(all_dice) if all_dice else 0:.4f}')
print(f'Successful (Dice > 0.01) mean: {np.mean(nonzero) if nonzero else 0:.4f} (n={len(nonzero)})')
print('='*40)
"

echo "Log: logs/val_job_${SLURM_JOB_ID}.out"
