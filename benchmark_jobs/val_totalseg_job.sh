#!/bin/bash
#SBATCH --job-name=val_totalseg
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:3
#SBATCH --time=12:00:00
#SBATCH --output=logs/val_totalseg_%j.out
#SBATCH --error=logs/val_totalseg_%j.err

cd "${SLURM_SUBMIT_DIR}"
[ ! -f .env ] && cd ..
source agentic/bin/activate
source .env

echo "[1/3] Cleanup..."
pkill -u $USER python; pkill -u $USER apptainer; sleep 2

echo "[2/3] Starting services..."
bash run_puhti.sh

echo "Waiting for service to be ready..."
until curl -s http://127.0.0.1:8011/health | grep -q "ok"; do
    sleep 10; echo "waiting for totalseg..."
done
echo "totalseg ready!"
sleep 10

echo "[3/3] Running validation (force_tool=totalseg)..."
DATASET_CSV="${VAL_DATA_DIR}/cases.csv"
OUTPUT_DIR="${VAL_DATA_DIR}/results_totalseg"
mkdir -p $OUTPUT_DIR

python -u validate_pipeline.py \
    --dataset_csv $DATASET_CSV \
    --output_dir $OUTPUT_DIR \
    --force_tool totalseg \
 2>&1

python -c "
import json, numpy as np, os
path = '$OUTPUT_DIR/results.jsonl'
results = [json.loads(l) for l in open(path)]
all_dice = [r['dice'] for r in results if r.get('dice') is not None]
nonzero = [d for d in all_dice if d > 0.01]
print('='*40)
print('Tool: totalseg')
print(f'Total: {len(results)}')
print(f'Successful: {len(nonzero)}/{len(results)}')
print(f'Mean Dice: {np.mean(all_dice) if all_dice else 0:.4f}')
print(f'Mean Dice>0.01: {np.mean(nonzero) if nonzero else 0:.4f}')
print('='*40)
"
