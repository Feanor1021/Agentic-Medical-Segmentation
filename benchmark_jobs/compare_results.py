#!/usr/bin/env python3
# =============================================================================
# compare_results.py
#
# Her tool'un validation sonuçlarını (results.jsonl) okuyup karşılaştırmalı
# bir tablo olarak yazdırır.
#
# Beklenen dizin yapısı
# ---------------------
#   BASE/results_totalseg/results.jsonl
#   BASE/results_voxtell/results.jsonl
#   BASE/results_biomedparse/results.jsonl
#   BASE/results/results.jsonl              (agentic pipeline)
#
# Her results.jsonl satırı: {"dice": 0.85, ...} formatında JSON.
#
# Çıktı sütunları
# ---------------
#   Cases   — toplam case sayısı
#   Success — Dice > 0.01 olan case sayısı / toplam
#   MeanDice — başarılı case'lerin ortalama Dice skoru
#   Median  — başarılı case'lerin median Dice skoru
#
# Kullanım
# --------
#   python compare_results.py
# =============================================================================

import json
import numpy as np
import os

BASE = "/scratch/project_2016517/furkan/val_data"

tools = ["totalseg", "voxtell", "biomedparse", "agentic"]

TOOL_PATHS = {
    "totalseg": "results_totalseg",
    "voxtell": "results_voxtell",
    "biomedparse": "results_biomedparse",
    "agentic": "results",
}

print("=" * 60)
print(f"{'Tool':<15} {'Cases':>6} {'Success':>10} {'MeanDice':>10} {'Median':>8}")
print("-" * 60)

for tool in tools:
    path = os.path.join(BASE, TOOL_PATHS.get(tool, f"results_{tool}"), "results.jsonl")
    if not os.path.exists(path):
        print(f"{tool:<15}  N/A")
        continue
    results = [json.loads(l) for l in open(path)]
    all_dice = [r["dice"] for r in results if r.get("dice") is not None]
    nonzero = [d for d in all_dice if d > 0.01]
    n = len(results)
    mean = np.mean(nonzero) if nonzero else 0
    median = float(np.median(nonzero)) if nonzero else 0
    print(f"{tool:<15} {n:>6} {len(nonzero):>5}/{n:<5} {mean:>10.4f} {median:>8.4f}")

print("=" * 60)