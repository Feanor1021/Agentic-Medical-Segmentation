#!/usr/bin/env python3
# =============================================================================
# download_msd_ct.py  —  MSD CT Dataset Downloader
#
# Sadece CT veri setlerini (Task03 Liver, Task09 Spleen, Task06 Lung) indirir
# and builds cases.csv with correct gt_label_value columns.
#
# Target distribution:
#   totalseg  liver       : 100 case  — Task03, gt_label_value=1
#   totalseg  spleen      : ~60 cases — Task09, gt_label_value=1 (all)
#   voxtell   liver_tumor : 100 case  — Task03, gt_label_value=2
#   voxtell   lung_tumor  : ~96 cases — Task06, gt_label_value=1 (all)
#
# GT FILES ARE NOT SPLIT:
#   Task03's original GT mask (label 1=liver, 2=tumor) is kept as-is.
#   The same image is added as both a "liver" and "liver_tumor" row in the CSV.
#   validate_pipeline reads gt_label_value to compute Dice on the correct label
#   (if label=2 is absent the tumour row is skipped).
#
# Usage:
#   python download_msd_ct.py --output_dir /scratch/.../val_data
#   python download_msd_ct.py --output_dir /scratch/.../val_data --no_label_check
#   python download_msd_ct.py --output_dir /scratch/.../val_data --skip_download
#   python download_msd_ct.py --output_dir /scratch/.../val_data --validate_only
# =============================================================================

import argparse
import csv
import os
import random
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import List, Tuple

MSD_BASE = "https://msd-for-monai.s3-us-west-2.amazonaws.com"

INSTRUCTIONS = {
    "liver": [
        "Segment liver",
        "Segment the liver",
        "Delineate the liver parenchyma",
        "Please segment the hepatic tissue",
        "Segment liver structure in this CT",
        "Identify and segment the liver in this CT scan",
    ],
    "liver_tumor": [
        "Segment liver tumor",
        "Segment hepatocellular carcinoma",
        "Find and segment liver lesion",
        "Segment the malignant liver mass",
        "Delineate liver neoplasm",
        "Segment cancerous region in liver",
        "Find and segment the hepatic tumor",
    ],
    "spleen": [
        "Segment spleen",
        "Segment the spleen",
        "Delineate the splenic tissue",
        "Please segment the spleen",
        "Identify and segment the spleen in this CT",
        "Segment splenic parenchyma",
    ],
    "lung_tumor": [
        "Segment lung tumor",
        "Segment pulmonary carcinoma",
        "Find and segment lung cancer",
        "Segment non-small cell lung cancer",
        "Delineate lung malignancy",
        "Segment lung neoplasm",
        "Find and segment the pulmonary tumor",
    ],
}


def pick(key: str) -> str:
    return random.choice(INSTRUCTIONS[key])


# ---------------------------------------------------------------------------
# Download / extract helpers
# ---------------------------------------------------------------------------

def is_valid_tar(path: str) -> bool:
    try:
        with tarfile.open(path) as t:
            t.getmembers()
        return True
    except Exception:
        return False


def wget_download(url: str, dest: str, desc: str = "") -> bool:
    """Download with wget; skip if already present and valid."""
    if os.path.exists(dest):
        if dest.endswith(".tar") and not is_valid_tar(dest):
            print(f"  [bozuk] {Path(dest).name} siliniyor, yeniden Downloading ...")
            os.remove(dest)
        else:
            print(f"  [skip]  {desc or Path(dest).name} zaten mevcut")
            return True
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  ⬇  {desc or url.split('/')[-1]} Downloading ...")
    r = subprocess.run(
        ["wget", "-q", "--show-progress", "--tries=3", "--retry-connrefused",
         "-O", dest, url],
        stderr=subprocess.STDOUT,
    )
    if r.returncode != 0:
        print(f"  ERROR: wget failed → {url}")
        if os.path.exists(dest):
            os.remove(dest)
        return False
    return True


def extract_tar(src: str, dst: str, max_cases: int = None):
    """Extract images + labels from a tar archive, then delete the tar."""
    os.makedirs(dst, exist_ok=True)
    print(f"  📦 Extracting {Path(src).name} ...")
    with tarfile.open(src) as t:
        members = t.getmembers()
        images = sorted(
            [m for m in members if "/imagesTr/" in m.name and m.name.endswith((".nii.gz", ".nii"))],
            key=lambda x: x.name,
        )
        labels = sorted(
            [m for m in members if "/labelsTr/" in m.name and m.name.endswith((".nii.gz", ".nii"))],
            key=lambda x: x.name,
        )
        others = [m for m in members if m.isdir() or "dataset.json" in m.name]
        if max_cases:
            images = images[:max_cases]
            labels = labels[:max_cases]
        t.extractall(dst, members=others + images + labels)
    os.remove(src)
    print(f"  🗑  Tar removed: {Path(src).name}")


def find_paired(image_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    """Return matched (image, label) pairs, skipping macOS junk files (._xxx)."""
    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        return []

    def valid_nifti(f: str) -> bool:
        return (f.endswith((".nii.gz", ".nii"))
                and not f.startswith("._")
                and not f.startswith("."))

    imgs = {
        Path(f).name.replace(".nii.gz", "").replace(".nii", ""): os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir)) if valid_nifti(f)
    }
    lbls = {
        Path(f).name.replace(".nii.gz", "").replace(".nii", ""): os.path.join(label_dir, f)
        for f in sorted(os.listdir(label_dir)) if valid_nifti(f)
    }
    return [(imgs[k], lbls[k]) for k in sorted(imgs) if k in lbls]


def has_label(lbl_path: str, val: int) -> bool:
    """Check whether a GT mask contains a specific label value."""
    try:
        import nibabel as nib
        import numpy as np
        arr = nib.load(lbl_path).get_fdata(dtype=np.float32)
        return val in np.unique(arr).astype(int)
    except Exception as e:
        print(f"    [warn] {Path(lbl_path).name}: {e}")
        return False


def download_task(base_dir: str, task: str, max_cases: int = None) -> List[Tuple[str, str]]:
    """Download and extract an MSD task, return (image, label) pairs."""
    task_dir = os.path.join(base_dir, task)
    tar_path = os.path.join(base_dir, f"{task}.tar")
    print(f"\n[{task}]")
    if not os.path.isdir(task_dir):
        for attempt in range(2):
            if not wget_download(f"{MSD_BASE}/{task}.tar", tar_path, task):
                return []
            try:
                extract_tar(tar_path, base_dir, max_cases=max_cases)
                break
            except Exception as e:
                print(f"  Extract error: {e}")
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                if attempt == 1:
                    return []
    else:
        print(f"  [skip]  {task} directory already exists")
    pairs = find_paired(
        os.path.join(task_dir, "imagesTr"),
        os.path.join(task_dir, "labelsTr"),
    )
    print(f"  ✓ {len(pairs)} pairs found")
    return pairs


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def build_liver_rows(
    pairs: List[Tuple[str, str]],
    liver_n: int,
    tumor_n: int,
    check_labels: bool = True,
) -> List[dict]:
    """Build CSV rows for Task03 Liver.

    The same (image, GT) pair is added as two rows:
      1. gt_label_value=1 → totalseg  → liver organ
      2. gt_label_value=2 → voxtell   → liver tumor (skipped if label=2 absent)
    validate_pipeline uses gt_label_value to compute Dice on the correct label.
    """
    rows_liver = []
    rows_tumor = []

    for img_path, lbl_path in pairs:
        if len(rows_liver) < liver_n:
            ok = has_label(lbl_path, 1) if check_labels else True
            if ok:
                rows_liver.append({
                    "case_id": f"ts_liver_{len(rows_liver):03d}",
                    "nifti_path": img_path,
                    "gt_mask_path": lbl_path,
                    "gt_label_value": 1,
                    "instruction": pick("liver"),
                    "tool": "totalseg",
                    "structure": "liver",
                    "modality": "CT",
                    "dataset": "MSD_Task03_Liver",
                })
        if len(rows_tumor) < tumor_n:
            ok = has_label(lbl_path, 2) if check_labels else True
            if ok:
                rows_tumor.append({
                    "case_id": f"vx_liver_tumor_{len(rows_tumor):03d}",
                    "nifti_path": img_path,
                    "gt_mask_path": lbl_path,
                    "gt_label_value": 2,
                    "instruction": pick("liver_tumor"),
                    "tool": "voxtell",
                    "structure": "liver_tumor",
                    "modality": "CT",
                    "dataset": "MSD_Task03_Liver",
                })
        if len(rows_liver) >= liver_n and len(rows_tumor) >= tumor_n:
            break

    print(f"  liver  (label=1) → totalseg : {len(rows_liver):>3} case")
    print(f"  tumor  (label=2) → voxtell  : {len(rows_tumor):>3} case")
    return rows_liver + rows_tumor


def build_spleen_rows(
    pairs: List[Tuple[str, str]],
    limit: int = 9999,
    check_labels: bool = True,
) -> List[dict]:
    """Task09 Spleen — single class (label=1), all cases."""
    rows = []
    for img_path, lbl_path in pairs:
        if len(rows) >= limit:
            break
        ok = has_label(lbl_path, 1) if check_labels else True
        if ok:
            rows.append({
                "case_id": f"ts_spleen_{len(rows):03d}",
                "nifti_path": img_path,
                "gt_mask_path": lbl_path,
                "gt_label_value": 1,
                "instruction": pick("spleen"),
                "tool": "totalseg",
                "structure": "spleen",
                "modality": "CT",
                "dataset": "MSD_Task09_Spleen",
            })
    print(f"  spleen (label=1) → totalseg : {len(rows):>3} case")
    return rows


def build_lung_rows(
    pairs: List[Tuple[str, str]],
    limit: int = 9999,
    check_labels: bool = True,
) -> List[dict]:
    """Task06 Lung — single class (label=1), all cases."""
    rows = []
    for img_path, lbl_path in pairs:
        if len(rows) >= limit:
            break
        ok = has_label(lbl_path, 1) if check_labels else True
        if ok:
            rows.append({
                "case_id": f"vx_lung_tumor_{len(rows):03d}",
                "nifti_path": img_path,
                "gt_mask_path": lbl_path,
                "gt_label_value": 1,
                "instruction": pick("lung_tumor"),
                "tool": "voxtell",
                "structure": "lung_tumor",
                "modality": "CT",
                "dataset": "MSD_Task06_Lung",
            })
    print(f"  lung   (label=1) → voxtell  : {len(rows):>3} case")
    return rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "case_id", "nifti_path", "gt_mask_path", "gt_label_value",
    "instruction", "tool", "structure", "modality", "dataset",
]


def write_csv(rows: List[dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ CSV written: {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_csv(csv_path: str) -> bool:
    """Check that all paths exist and all tools/label values are valid."""
    print(f"\n{'='*60}")
    print(f"CSV Validation: {csv_path}")
    print(f"{'='*60}")

    VALID_TOOLS = {"totalseg", "voxtell", "biomedparse"}
    errors: List[str] = []
    warnings: List[str] = []
    tool_c: dict = {}
    struct_c: dict = {}
    ds_c: dict = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows, 1):
        cid = row.get("case_id", f"row_{i}")
        nifti = row.get("nifti_path", "").strip()
        gt = row.get("gt_mask_path", "").strip()
        tool = row.get("tool", "").strip()
        struc = row.get("structure", "").strip()
        gt_lv = row.get("gt_label_value", "").strip()

        if not nifti:
            errors.append(f"[{cid}] nifti_path is empty")
        elif not os.path.exists(nifti):
            errors.append(f"[{cid}] nifti_path yok: {nifti}")

        if not gt:
            warnings.append(f"[{cid}] gt_mask_path is empty")
        elif not os.path.exists(gt):
            errors.append(f"[{cid}] gt_mask_path yok: {gt}")

        if tool not in VALID_TOOLS:
            errors.append(f"[{cid}] invalid tool: '{tool}'")

        if gt_lv and not gt_lv.isdigit():
            errors.append(f"[{cid}] gt_label_value is not an integer: '{gt_lv}'")

        tool_c[tool] = tool_c.get(tool, 0) + 1
        struct_c[struc or "(empty)"] = struct_c.get(struc or "(empty)", 0) + 1
        ds_c[row.get("dataset", "")] = ds_c.get(row.get("dataset", ""), 0) + 1

    print(f"\nTotal  : {len(rows)}")
    print(f"\nTool distribution:")
    for t, c in sorted(tool_c.items()):
        print(f"  {t:<15}: {c:>4}")
    print(f"\nStructure distribution:")
    for s, c in sorted(struct_c.items()):
        print(f"  {s:<22}: {c:>4}")
    print(f"\nDataset distribution:")
    for d, c in sorted(ds_c.items()):
        print(f"  {d:<30}: {c:>4}")

    if warnings:
        print(f"\n⚠  {len(warnings)} warning(s)")
        for w in warnings[:10]:
            print(f"   {w}")

    if errors:
        print(f"\n❌ {len(errors)} hata")
        for e in errors[:20]:
            print(f"   {e}")
        print("\n❌ Validation FAILED")
        return False

    print(f"\n✅ Validation passed")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download MSD Task03+Task09+Task06 and build cases.csv")
    parser.add_argument("--output_dir", default="/scratch/project_2016517/furkan/val_data")
    parser.add_argument("--liver_n", type=int, default=100, help="Number of liver organ (label=1) cases")
    parser.add_argument("--liver_tumor_n", type=int, default=100, help="Number of liver tumour (label=2) cases")
    parser.add_argument("--no_label_check", action="store_true",
                        help="Skip nibabel label check — faster but tumour-free cases may be included")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, only rebuild CSV from existing files")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate the existing cases.csv without downloading")
    parser.add_argument("--csv_name", default="cases.csv")
    args = parser.parse_args()

    base_dir = args.output_dir
    csv_path = os.path.join(base_dir, args.csv_name)
    os.makedirs(base_dir, exist_ok=True)
    check = not args.no_label_check

    if args.validate_only:
        if not os.path.exists(csv_path):
            print(f"❌ CSV not found: {csv_path}")
            sys.exit(1)
        sys.exit(0 if validate_csv(csv_path) else 1)

    if not args.skip_download:
        liver_pairs = download_task(base_dir, "Task03_Liver")
        spleen_pairs = download_task(base_dir, "Task09_Spleen")
        lung_pairs = download_task(base_dir, "Task06_Lung")
    else:
        print("\n[skip_download] Collecting pairs from existing files...")
        liver_pairs = find_paired(f"{base_dir}/Task03_Liver/imagesTr", f"{base_dir}/Task03_Liver/labelsTr")
        spleen_pairs = find_paired(f"{base_dir}/Task09_Spleen/imagesTr", f"{base_dir}/Task09_Spleen/labelsTr")
        lung_pairs = find_paired(f"{base_dir}/Task06_Lung/imagesTr", f"{base_dir}/Task06_Lung/labelsTr")
        print(f"  Task03 Liver:  {len(liver_pairs)} pairs")
        print(f"  Task09 Spleen: {len(spleen_pairs)} pairs")
        print(f"  Task06 Lung:   {len(lung_pairs)} pairs")

    print(f"\n{'='*60}")
    print("Building rows...")
    print(f"{'='*60}")
    if check:
        print("ℹ  Label check ENABLED — slower but safe\n")
    else:
        print("⚡ Label check DISABLED (--no_label_check)\n")

    rows = []
    print(f"Task03 Liver  → liver:{args.liver_n}  liver_tumor:{args.liver_tumor_n}")
    rows += build_liver_rows(liver_pairs, args.liver_n, args.liver_tumor_n, check)
    print(f"\nTask09 Spleen → all cases")
    rows += build_spleen_rows(spleen_pairs, limit=9999, check_labels=check)
    print(f"\nTask06 Lung   → all cases")
    rows += build_lung_rows(lung_pairs, limit=9999, check_labels=check)

    tc = Counter(r["tool"] for r in rows)
    sc = Counter(r["structure"] for r in rows)
    print(f"\n{'='*60}")
    print(f"SUMMARY — Total {len(rows)} cases")
    print(f"  totalseg : {tc.get('totalseg', 0):>4}  (liver:{sc.get('liver', 0)}  spleen:{sc.get('spleen', 0)})")
    print(f"  voxtell  : {tc.get('voxtell', 0):>4}  (liver_tumor:{sc.get('liver_tumor', 0)}  lung_tumor:{sc.get('lung_tumor', 0)})")
    print(f"{'='*60}")

    write_csv(rows, csv_path)
    ok = validate_csv(csv_path)

    print(f"\nNext steps:")
    print(f"  python validate_pipeline.py --dataset_csv {csv_path} --output_dir {base_dir}/results")
    print(f"  python validate_pipeline.py --dataset_csv {csv_path} --output_dir {base_dir}/results_totalseg --force_tool totalseg --tool_filter totalseg")
    print(f"  python validate_pipeline.py --dataset_csv {csv_path} --output_dir {base_dir}/results_voxtell  --force_tool voxtell  --tool_filter voxtell")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()