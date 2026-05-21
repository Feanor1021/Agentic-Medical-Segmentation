#!/bin/bash
# =============================================================================
# val_job.sh
#
# Puhti SLURM job: tam agentic pipeline validation (VLM + Planner + Critic).
# Tüm servisleri başlatır, health check geçince validation'ı çalıştırır.
#
# Adımlar:
#   1. Eski servisleri temizle
#   2. run_puhti.sh ile tüm servisleri başlat
#   3. Tüm servisler health check geçene kadar bekle (port: 8001-8013)
#   4. validate_pipeline.py çalıştır (agent modu — force_tool yok)
#   5. Dice özeti yazdır
# =============================================================================
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

cd /scratch/project_2016517/furkan/agentic-seg
source agentic/bin/activate

echo "[1/4] Eski servisler temizleniyor..."
pkill -u $USER python
pkill -u $USER apptainer

pip install -q numpy nibabel requests scipy

echo "[2/4] Servisler başlatılıyor (run_puhti.sh)..."
bash run_puhti.sh
echo "Servislerin ayağa kalkması bekleniyor (90s)..."

echo "Servisler bekleniyor..."
for port in 8001 8002 8011 8012 8013; do
    until curl -s http://127.0.0.1:$port/health | grep -q "ok"; do
        sleep 10; echo "Port $port bekleniyor..."
    done
    echo "Port $port hazir!"
done
echo "Tum servisler hazir!"

echo "[3/4] Validation başlatılıyor..."
DATASET_CSV="/scratch/project_2016517/furkan/val_data/cases.csv"
OUTPUT_DIR="/scratch/project_2016517/furkan/val_data/results"

python -u validate_pipeline.py \
    --dataset_csv $DATASET_CSV \
    --output_dir $OUTPUT_DIR 2>&1

echo "[4/4] İşlem tamamlandı. Özet Sonuçlar:"
python -c "
import json, numpy as np, os
path = '$OUTPUT_DIR/results.jsonl'
if not os.path.exists(path):
    print('HATA: Sonuç dosyası bulunamadı!')
    exit()
results = [json.loads(l) for l in open(path)]
all_dice = [r['dice'] for r in results if r['dice'] is not None]
nonzero = [d for d in all_dice if d > 0.01]
print('='*40)
print(f'Toplam Case  : {len(results)}')
print(f'Ortalama Dice: {np.mean(all_dice) if all_dice else 0:.4f}')
print(f'Başarılı (Dice > 0.01) Ortalaması: {np.mean(nonzero) if nonzero else 0:.4f} (n={len(nonzero)})')
print('='*40)
"

echo "Log dosyası: logs/val_job_${SLURM_JOB_ID}.out"