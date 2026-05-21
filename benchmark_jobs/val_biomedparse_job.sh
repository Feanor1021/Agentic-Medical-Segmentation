#!/bin/bash
# =============================================================================
# val_biomedparse_job.sh
#
# Puhti SLURM job: BiomedParse servisini ayağa kaldırır ve validation çalıştırır.
#
# Adımlar:
#   1. Çalışan python/apptainer proseslerini temizler
#   2. run_puhti.sh ile servisleri başlatır
#   3. BiomedParse health check geçene kadar bekler
#   4. validate_pipeline.py'yi force_tool=biomedparse ile çalıştırır
#   5. Sonuçları (Dice skoru, başarı oranı) özetler
# =============================================================================
#SBATCH --job-name=val_biomedparse
#SBATCH --account=project_2016517
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:3
#SBATCH --time=12:00:00
#SBATCH --output=logs/val_biomedparse_%j.out
#SBATCH --error=logs/val_biomedparse_%j.err

cd /scratch/project_2016517/furkan/agentic-seg
source agentic/bin/activate

echo "[1/3] Temizlik..."
pkill -u $USER python; pkill -u $USER apptainer; sleep 2

echo "[2/3] Servisler baslatiliyor..."
bash run_puhti.sh

echo "Servis hazir olana kadar bekleniyor..."
until curl -s http://127.0.0.1:8013/health | grep -q "ok"; do
    sleep 10; echo "biomedparse bekleniyor..."
done
echo "biomedparse hazir!"
sleep 10

echo "[3/3] Validation (force_tool=biomedparse)..."
DATASET_CSV="/scratch/project_2016517/furkan/val_data/cases.csv"
OUTPUT_DIR="/scratch/project_2016517/furkan/val_data/results_biomedparse"
mkdir -p $OUTPUT_DIR

python -u validate_pipeline.py \
    --dataset_csv $DATASET_CSV \
    --output_dir $OUTPUT_DIR \
    --force_tool biomedparse \
 2>&1

# Sonuçları özetle: toplam case, başarılı segmentasyon sayısı, mean/nonzero Dice
python -c "
import json, numpy as np, os
path = '$OUTPUT_DIR/results.jsonl'
results = [json.loads(l) for l in open(path)]
all_dice = [r['dice'] for r in results if r.get('dice') is not None]
nonzero = [d for d in all_dice if d > 0.01]
print('='*40)
print('Tool: biomedparse')
print(f'Toplam: {len(results)}')
print(f'Basarili: {len(nonzero)}/{len(results)}')
print(f'Mean Dice: {np.mean(all_dice) if all_dice else 0:.4f}')
print(f'Mean Dice>0.01: {np.mean(nonzero) if nonzero else 0:.4f}')
print('='*40)
"