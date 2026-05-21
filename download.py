#!/usr/bin/env python3
# =============================================================================
# download.py  —  Validation Dataset Downloader (wget tabanlı)
#
# MSD (Medical Segmentation Decathlon) ve diğer public veri setlerini
# indirir, cases.csv'yi oluşturur.
#
# Desteklenen veri setleri:
#   TotalSeg  : Task03 Liver (organ), Task09 Spleen
#   VoxTell   : Task03 Liver (tumor), Task06 Lung tumor, Task07 Pancreas tumor,
#               Task01 Brain tumor, Task10 Colon, COVID-19 CT
#   BiomedParse: COVID-19 CT, CHAOS MRI, Task05 Prostate, Task04 Hippocampus,
#                Task02 Heart
#
# Çalıştırma:
#   python3 download.py --output_dir /scratch/.../val_data
#   python3 download.py --output_dir /scratch/.../val_data --total_cases 500
# =============================================================================

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional

MSD_BASE = "https://msd-for-monai.s3-us-west-2.amazonaws.com"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def is_valid_tar(path: str) -> bool:
    try:
        with tarfile.open(path) as t:
            t.getmembers()
        return True
    except Exception:
        return False


def wget_download(url: str, dest: str, desc: str = "") -> bool:
    """Download a file with wget; skip if already present and valid."""
    if os.path.exists(dest):
        if dest.endswith(".tar") and not is_valid_tar(dest):
            print(f"  [bozuk] {Path(dest).name} siliniyor, yeniden indiriliyor...")
            os.remove(dest)
        else:
            print(f"  [skip] {desc or Path(dest).name} zaten var")
            return True
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  Indiriliyor: {desc or url.split('/')[-1]} ...")
    r = subprocess.run(
        ["wget", "-q", "--show-progress", "-O", dest, url],
        stderr=subprocess.STDOUT,
    )
    if r.returncode != 0:
        print(f"  HATA: wget failed for {url}")
        if os.path.exists(dest):
            os.remove(dest)
        return False
    return True


def extract_tar(src: str, dst: str, max_images: int = None, max_labels: int = None):
    """Extract up to max_images image+label pairs from a tar, then delete the tar."""
    os.makedirs(dst, exist_ok=True)
    print(f"  Extracting {Path(src).name} ...")
    with tarfile.open(src) as t:
        members = t.getmembers()
        if max_images:
            images = sorted([m for m in members if "/imagesTr/" in m.name and m.name.endswith((".nii.gz", ".nii"))], key=lambda x: x.name)
            labels = sorted([m for m in members if "/labelsTr/" in m.name and m.name.endswith((".nii.gz", ".nii"))], key=lambda x: x.name)
            others = [m for m in members if m.isdir() or "dataset.json" in m.name]
            selected = others + images[:max_images] + labels[:max_images]
            t.extractall(dst, members=selected)
        else:
            t.extractall(dst)
    os.remove(src)
    print(f"  Tar silindi: {Path(src).name}")


def extract_zip(src: str, dst: str):
    os.makedirs(dst, exist_ok=True)
    print(f"  Extracting {Path(src).name} ...")
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)
    os.remove(src)
    print(f"  Zip silindi: {Path(src).name}")


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------

def list_niftis(folder: str, max_n: int) -> List[str]:
    """Recursively find up to max_n NIfTI files under folder."""
    found = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.endswith(".nii.gz") or f.endswith(".nii"):
                found.append(os.path.join(root, f))
                if len(found) >= max_n:
                    return found
    return found


def find_paired(image_dir: str, label_dir: str, max_n: int) -> List[Tuple[str, str]]:
    """Return matched (image, label) pairs from two directories."""
    pairs = []
    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        return pairs
    img = {Path(f).name.replace(".nii.gz", "").replace(".nii", ""): os.path.join(image_dir, f)
           for f in sorted(os.listdir(image_dir)) if f.endswith((".nii.gz", ".nii"))}
    lbl = {Path(f).name.replace(".nii.gz", "").replace(".nii", ""): os.path.join(label_dir, f)
           for f in sorted(os.listdir(label_dir)) if f.endswith((".nii.gz", ".nii"))}
    for k in sorted(img):
        if k in lbl:
            pairs.append((img[k], lbl[k]))
            if len(pairs) >= max_n:
                break
    return pairs


# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

TEMPLATES = {
    "liver": [
        "Segment liver", "Segment the liver", "Delineate the liver parenchyma",
        "Please segment the hepatic tissue", "Segment liver structure in this CT",
    ],
    "spleen": [
        "Segment spleen", "Segment the spleen", "Delineate the splenic tissue",
        "Please segment the spleen", "Identify and segment the spleen",
    ],
    "kidney_left": [
        "Segment left kidney", "Segment the left renal organ",
        "Delineate the left kidney", "Identify the left kidney and segment it",
    ],
    "kidney_right": [
        "Segment right kidney", "Segment the right renal organ",
        "Delineate the right kidney", "Identify the right kidney and segment it",
    ],
    "kidney_left,kidney_right": [
        "Segment both kidneys", "Segment left and right kidneys",
        "Delineate bilateral kidneys", "Identify and segment kidneys",
    ],
    "pancreas": [
        "Segment pancreas", "Segment the pancreatic gland",
        "Delineate the pancreas", "Please segment pancreatic tissue",
    ],
    "liver_tumor": [
        "Segment liver tumor", "Segment hepatocellular carcinoma",
        "Find and segment liver lesion", "Segment the malignant liver mass",
        "Delineate liver neoplasm", "Segment cancerous region in liver",
    ],
    "lung_tumor": [
        "Segment lung tumor", "Segment pulmonary carcinoma",
        "Find and segment lung cancer", "Segment non-small cell lung cancer",
        "Delineate lung malignancy", "Segment lung neoplasm",
    ],
    "pancreas_tumor": [
        "Segment pancreatic tumor", "Segment pancreatic ductal adenocarcinoma",
        "Find and segment pancreatic lesion", "Segment pancreatic cancer",
    ],
    "kidney_tumor": [
        "Segment kidney tumor", "Segment renal cell carcinoma",
        "Find and segment renal tumor", "Segment renal neoplasm",
    ],
    "brain_tumor": [
        "Segment brain tumor", "Segment glioblastoma",
        "Delineate brain lesion", "Find and segment brain neoplasm",
    ],
    "colon_cancer": [
        "Segment colon cancer", "Segment colorectal carcinoma",
        "Find and segment colon tumor", "Delineate colon neoplasm",
    ],
    "lung_infection": [
        "Segment lung infection", "Segment COVID-19 lung lesions",
        "Segment ground-glass opacities in lung", "Delineate pneumonia regions in CT",
        "Segment pulmonary infection", "Find and segment GGO in lung CT",
    ],
    "liver_mri": [
        "Segment liver in MRI", "Segment the liver on this MRI scan",
        "Delineate liver in T2-weighted MRI", "Segment liver parenchyma in MRI",
    ],
}


def _guess_key(case_id: str, structure: str) -> str:
    s = (case_id + " " + structure).lower()
    if "covid" in s or "ggo" in s or "infection" in s:       return "lung_infection"
    if "liver" in s and ("tumor" in s or "lesion" in s):     return "liver_tumor"
    if "lung" in s and ("tumor" in s or "cancer" in s):      return "lung_tumor"
    if "pancreas" in s and ("tumor" in s or "cancer" in s):  return "pancreas_tumor"
    if "kidney" in s and ("tumor" in s or "cancer" in s):    return "kidney_tumor"
    if "brain" in s and ("tumor" in s or "lesion" in s):     return "brain_tumor"
    if "colon" in s and ("cancer" in s or "tumor" in s):     return "colon_cancer"
    if "liver" in s and "mri" in s:                          return "liver_mri"
    if "liver" in s:   return "liver"
    if "spleen" in s:  return "spleen"
    if "kidney_left" in s and "kidney_right" in s: return "kidney_left,kidney_right"
    if "kidney_left" in s:  return "kidney_left"
    if "kidney_right" in s: return "kidney_right"
    if "pancreas" in s: return "pancreas"
    return ""


def pick_instruction(case_id: str, structure: str, fallback: str = "") -> str:
    key = structure.strip() if structure.strip() in TEMPLATES else _guess_key(case_id, structure)
    if key in TEMPLATES:
        return random.choice(TEMPLATES[key])
    return fallback or f"Segment {structure or 'target structure'}"


def make_row(case_id, nifti, gt, tool, structure="", instruction=""):
    if not instruction:
        instruction = pick_instruction(case_id, structure)
    return {
        "case_id": case_id, "nifti_path": nifti, "gt_mask_path": gt,
        "instruction": instruction, "tool": tool, "structure": structure,
    }


# ---------------------------------------------------------------------------
# Dataset downloaders — all from AWS S3 msd-for-monai
# ---------------------------------------------------------------------------

def download_msd(base_dir, task, n_rows, row_fn) -> List[dict]:
    """Generic MSD task downloader."""
    task_dir = os.path.join(base_dir, task)
    tar_path = os.path.join(base_dir, f"{task}.tar")
    print(f"\n[{task}]")
    if not os.path.isdir(task_dir):
        for attempt in range(2):
            ok = wget_download(f"{MSD_BASE}/{task}.tar", tar_path, task)
            if not ok:
                return []
            try:
                extract_tar(tar_path, base_dir, max_images=150)
                break
            except Exception as e:
                print(f"  Extract hatasi: {e} — tar bozuk, siliniyor...")
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                if attempt == 1:
                    print(f"  SKIP: {task}")
                    return []
    img_dir = os.path.join(task_dir, "imagesTr")
    lbl_dir = os.path.join(task_dir, "labelsTr")
    pairs = find_paired(img_dir, lbl_dir, n_rows)
    print(f"  {len(pairs)} case bulundu")
    return [row_fn(i, img, lbl) for i, (img, lbl) in enumerate(pairs)]


def get_task03(base_dir, n_ts, n_vx):
    """Task03 Liver — totalseg(liver) + voxtell(liver tumor)."""
    all_rows = download_msd(base_dir, "Task03_Liver", n_ts + n_vx,
                            lambda i, img, lbl: make_row(
                                f"{'ts_liver' if i < n_ts else 'vx_liver_tumor'}_{i if i < n_ts else i - n_ts:03d}",
                                img, lbl,
                                "totalseg" if i < n_ts else "voxtell",
                                "liver" if i < n_ts else ""))
    return all_rows


def get_task06(base_dir, n):
    """Task06 Lung — voxtell."""
    return download_msd(base_dir, "Task06_Lung", n,
                        lambda i, img, lbl: make_row(f"vx_lung_tumor_{i:03d}", img, lbl, "voxtell"))


def get_task07(base_dir, n):
    """Task07 Pancreas — voxtell."""
    return download_msd(base_dir, "Task07_Pancreas", n,
                        lambda i, img, lbl: make_row(f"vx_pancreas_tumor_{i:03d}", img, lbl, "voxtell"))


def get_task09(base_dir, n):
    """Task09 Spleen — totalseg."""
    return download_msd(base_dir, "Task09_Spleen", n,
                        lambda i, img, lbl: make_row(f"ts_spleen_{i:03d}", img, lbl, "totalseg", "spleen"))


def get_task01(base_dir, n):
    """Task01 Brain Tumor — voxtell."""
    return download_msd(base_dir, "Task01_BrainTumour", n,
                        lambda i, img, lbl: make_row(f"vx_brain_tumor_{i:03d}", img, lbl, "voxtell"))


def get_task10(base_dir, n):
    """Task10 Colon — voxtell."""
    return download_msd(base_dir, "Task10_Colon", n,
                        lambda i, img, lbl: make_row(f"vx_colon_cancer_{i:03d}", img, lbl, "voxtell"))


def get_covid(base_dir, n_vx, n_bp):
    """COVID-19 CT — voxtell (n_vx) + biomedparse (n_bp)."""
    covid_dir = os.path.join(base_dir, "COVID-19-CT-Seg")
    zip_path = os.path.join(base_dir, "COVID-19-CT-Seg.zip")
    print("\n[COVID-19-CT-Seg]")
    if not os.path.isdir(covid_dir):
        ok = wget_download(
            "https://zenodo.org/records/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1",
            zip_path, "COVID-19-CT-Seg")
        if not ok:
            return []
        extract_zip(zip_path, covid_dir)
    niftis = list_niftis(covid_dir, n_vx + n_bp)
    print(f"  {len(niftis)} case bulundu")
    rows = []
    for i, nf in enumerate(niftis[:n_vx]):
        rows.append(make_row(f"vx_covid_{i:03d}", nf, "", "voxtell"))
    for i, nf in enumerate(niftis[n_vx:n_vx + n_bp]):
        rows.append(make_row(f"bp_covid_{i:03d}", nf, "", "biomedparse"))
    return rows


def get_chaos(base_dir, n):
    """CHAOS MRI — biomedparse."""
    chaos_dir = os.path.join(base_dir, "CHAOS")
    zip_path = os.path.join(base_dir, "CHAOS_Train_Sets.zip")
    print("\n[CHAOS MRI]")
    if not os.path.isdir(chaos_dir):
        ok = wget_download(
            "https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1",
            zip_path, "CHAOS MRI")
        if not ok:
            return []
        extract_zip(zip_path, chaos_dir)
    niftis = list_niftis(chaos_dir, n)
    print(f"  {len(niftis)} case bulundu")
    return [make_row(f"bp_chaos_{i:03d}", nf, "", "biomedparse", "", "Segment liver in MRI")
            for i, nf in enumerate(niftis)]


def get_task05(base_dir, n):
    """Task05 Prostate MRI — biomedparse."""
    return download_msd(base_dir, "Task05_Prostate", n,
                        lambda i, img, lbl: make_row(f"bp_prostate_{i:03d}", img, lbl, "biomedparse", "",
                                                     random.choice([
                                                         "Segment prostate in MRI",
                                                         "Segment the prostate gland in MRI",
                                                         "Delineate prostate in T2-weighted MRI",
                                                         "Please segment prostate tissue in MRI",
                                                         "Find and segment the prostate",
                                                     ])))


def get_task04(base_dir, n):
    """Task04 Hippocampus MRI — biomedparse."""
    return download_msd(base_dir, "Task04_Hippocampus", n,
                        lambda i, img, lbl: make_row(f"bp_hippocampus_{i:03d}", img, lbl, "biomedparse", "",
                                                     random.choice([
                                                         "Segment hippocampus in MRI",
                                                         "Segment the hippocampus in brain MRI",
                                                         "Delineate hippocampus in T1-weighted MRI",
                                                         "Please segment hippocampal structure in MRI",
                                                     ])))


def get_task02(base_dir, n):
    """Task02 Heart MRI — biomedparse."""
    return download_msd(base_dir, "Task02_Heart", n,
                        lambda i, img, lbl: make_row(f"bp_heart_{i:03d}", img, lbl, "biomedparse", "",
                                                     random.choice([
                                                         "Segment heart in MRI",
                                                         "Segment the left ventricle in cardiac MRI",
                                                         "Delineate cardiac structure in MRI",
                                                         "Please segment myocardium in MRI",
                                                     ])))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/scratch/project_2016517/furkan/val_data")
    parser.add_argument("--total_cases", type=int, default=1000)
    args = parser.parse_args()

    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    print(f"Hedef: {args.total_cases} case | Output: {base_dir}\n")

    rows = []
    rows += get_task03(base_dir, n_ts=100, n_vx=50)   # 150 (100 liver organ + 50 liver tumor)
    rows += get_task09(base_dir, n=100)                # 100 spleen
    rows += get_task06(base_dir, n=100)                # 100 lung tumor
    rows += get_task07(base_dir, n=100)                # 100 pancreas tumor
    rows += get_task01(base_dir, n=80)                 # 80 brain tumor
    rows += get_task10(base_dir, n=70)                 # 70 colon cancer
    rows += get_covid(base_dir, n_vx=60, n_bp=60)     # 60 voxtell + 60 biomedparse
    rows += get_task05(base_dir, n=100)                # 100 prostate MRI
    rows += get_task04(base_dir, n=100)                # 100 hippocampus MRI
    rows += get_task02(base_dir, n=50)                 # 50 heart MRI
    rows += get_chaos(base_dir, n=100)                 # 100 abdominal MRI

    from collections import Counter
    counts = Counter(r["tool"] for r in rows)
    print(f"\nToplam {len(rows)} case:")
    for t, c in counts.items():
        print(f"  {t:15s}: {c}")

    csv_path = os.path.join(base_dir, "cases.csv")
    fields = ["case_id", "nifti_path", "gt_mask_path", "instruction", "tool", "structure"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nCSV: {csv_path}")
    print(f"\nSonraki adim:")
    print(f"  python3 validate_pipeline.py --dataset_csv {csv_path} --output_dir {base_dir}/results")


if __name__ == "__main__":
    main()