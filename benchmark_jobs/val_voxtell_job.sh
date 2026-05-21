#!/bin/bash
# =============================================================================
# val_voxtell_job.sh
#
# Puhti SLURM job: VoxTell servisini ayağa kaldırır ve validation çalıştırır.
#
# Adımlar:
#   1. Çalışan python/apptainer proseslerini temizler
#   2. run_puhti.sh ile servisleri başlatır
#   3. VoxTell health check geçene kadar bekler
#   4. validate_pipeline.py'yi force_tool=voxtell ile çalıştırır
#   5. Sonuçları (Dice skoru, başarı oranı) özetler
# =============================================================================
#SBATCH --job-name=val_voxtell
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:3
#SBATCH --time=12:00:00
#SBATCH --output=logs/val_voxtell_%j.out
#SBATCH --error=logs/val_voxtell_%j.err

cd /scratch/project_2016517/furkan/agentic-seg
source agentic/bin/activate

echo "[1/3] Temizlik..."
pkill -u $USER python; pkill -u $USER apptainer; sleep 2

echo "[2/3] Servisler baslatiliyor..."
bash run_puhti.sh

echo "Servis hazir olana kadar bekleniyor..."
until curl -s http://127.0.0.1:8012/health | grep -q "ok"; do
    sleep 10; echo "voxtell bekleniyor..."
done
echo "voxtell hazir!"
sleep 10

echo "[3/3] Validation (force_tool=voxtell)..."
DATASET_CSV="/scratch/project_2016517/furkan/val_data/cases.csv"
OUTPUT_DIR="/scratch/project_2016517/furkan/val_data/results_voxtell"
mkdir -p $OUTPUT_DIR

python -u validate_pipeline.py \
    --dataset_csv $DATASET_CSV \
    --output_dir $OUTPUT_DIR \
    --force_tool voxtell \
 2>&1

# Sonuçları özetle: toplam case, başarılı segmentasyon sayısı, mean/nonzero Dice
python -c "
import json, numpy as np, os
path = '$OUTPUT_DIR/results.jsonl'
results = [json.loads(l) for l in open(path)]
all_dice = [r['dice'] for r in results if r.get('dice') is not None]
nonzero = [d for d in all_dice if d > 0.01]
print('='*40)
print('Tool: voxtell')
print(f'Toplam: {len(results)}')
print(f'Basarili: {len(nonzero)}/{len(results)}')
print(f'Mean Dice: {np.mean(all_dice) if all_dice else 0:.4f}')
print(f'Mean Dice>0.01: {np.mean(nonzero) if nonzero else 0:.4f}')
print('='*40)
"