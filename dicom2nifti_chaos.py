#!/usr/bin/env python3
# =============================================================================
# dicom2nifti_chaos.py  —  Genel DICOM → NIfTI Converter
#
# Finds all DICOM directories under the scan root, converts them to NIfTI,
# and generates rows ready for cases.csv.
#
# Also scans Ground directories in the CHAOS dataset: PNG ground-truth
# masks are converted to NIfTI if present.
#
# Usage:
#   python3 dicom2nifti_chaos.py \
#       --scan_dir  /scratch/.../val_data \
#       --output_dir /scratch/.../val_data/converted_nifti \
#       --csv_out   /scratch/.../val_data/converted_cases.csv
#
# Arguments:
#   --scan_dir   : root directory to scan
#   --output_dir : directory for converted NIfTI files
#   --csv_out    : output CSV file path
#   --append_to  : append output rows to an existing CSV
# =============================================================================

import argparse
import csv
import os
import sys
import random
import numpy as np
import nibabel as nib


def is_dicom_dir(path):
    """Return True if the directory contains at least one DICOM file."""
    if not os.path.isdir(path):
        return False
    for f in os.listdir(path):
        if f.startswith('.') or f.startswith('._'):
            continue
        fp = os.path.join(path, f)
        if os.path.isfile(fp) and (f.endswith('.dcm') or f.endswith('.DCM')):
            return True
        # Extension-less DICOM files
        if os.path.isfile(fp) and '.' not in f:
            try:
                import pydicom
                pydicom.dcmread(fp, stop_before_pixels=True)
                return True
            except Exception:
                pass
    return False


def find_all_dicom_dirs(root):
    """Recursively find all DICOM directories under root."""
    dicom_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if 'converted_nifti' in dirpath or 'CHAOS_nifti' in dirpath:
            continue
        if '/results' in dirpath:
            continue
        if is_dicom_dir(dirpath):
            dicom_dirs.append(dirpath)
    return dicom_dirs


def convert_dicom_series(dicom_dir, out_path):
    """Convert a DICOM series directory to a NIfTI file."""
    try:
        import pydicom
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pydicom'], check=True)
        import pydicom

    files = sorted([
        os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
        if not f.startswith('.') and not f.startswith('._')
        and os.path.isfile(os.path.join(dicom_dir, f))
    ])
    if not files:
        return False

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        return False

    slices.sort(key=lambda x: float(getattr(x, 'InstanceNumber', 0)))
    vol = np.stack([s.pixel_array for s in slices], axis=-1).astype(np.float32)

    ds = slices[0]
    ps = getattr(ds, 'PixelSpacing', [1.0, 1.0])
    st = float(getattr(ds, 'SliceThickness', 1.0) or 1.0)
    affine = np.diag([float(ps[0]), float(ps[1]), st, 1.0])

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    nib.save(nib.Nifti1Image(vol, affine), out_path)
    return True


def convert_ground_png(ground_dir, out_path):
    """Convert CHAOS-style PNG ground-truth masks to a NIfTI binary mask."""
    try:
        from PIL import Image
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'Pillow'], check=True)
        from PIL import Image

    files = sorted([
        os.path.join(ground_dir, f) for f in os.listdir(ground_dir)
        if f.lower().endswith('.png') and not f.startswith('.')
    ])
    if not files:
        return False

    slices = []
    for f in files:
        try:
            img = np.array(Image.open(f).convert('L'))
            slices.append((img > 0).astype(np.uint8))
        except Exception:
            continue

    if not slices:
        return False

    vol = np.stack(slices, axis=-1)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), out_path)
    return True


def guess_modality_instruction(dicom_dir, case_id):
    """Infer modality and pick a matching instruction from the path/case_id."""
    path_lower = dicom_dir.lower()
    if 't2' in path_lower or 'mri' in path_lower or 't1' in path_lower:
        modality = 'MRI'
        instructions = [
            "Segment liver in MRI",
            "Segment the liver on this MRI scan",
            "Delineate liver in MRI",
            "Segment abdominal organ in MRI",
        ]
    elif 'ct' in path_lower:
        modality = 'CT'
        instructions = [
            "Segment liver in CT",
            "Segment abdominal organ",
            "Delineate organ boundaries",
        ]
    else:
        modality = 'MRI'
        instructions = [
            "Segment the target structure",
            "Segment organ in this medical image",
        ]
    return modality, random.choice(instructions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_dir', default='/scratch/project_2016517/furkan/val_data')
    parser.add_argument('--output_dir', default='/scratch/project_2016517/furkan/val_data/converted_nifti')
    parser.add_argument('--csv_out', default='/scratch/project_2016517/furkan/val_data/converted_cases.csv')
    parser.add_argument('--append_to', default='', help='Mevcut CSV dosyasina ekle')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Taraniyor: {args.scan_dir}")
    dicom_dirs = find_all_dicom_dirs(args.scan_dir)
    print(f"Bulunan DICOM klasoru: {len(dicom_dirs)}")

    rows = []
    for i, dicom_dir in enumerate(dicom_dirs):
        rel = os.path.relpath(dicom_dir, args.scan_dir)
        case_id = "dc_" + rel.replace(os.sep, '_').replace(' ', '_').lower()
        case_id = ''.join(c if c.isalnum() or c == '_' else '_' for c in case_id)[:60]

        nifti_out = os.path.join(args.output_dir, f"{case_id}.nii.gz")
        gt_out = os.path.join(args.output_dir, f"{case_id}_gt.nii.gz")

        print(f"[{i+1}/{len(dicom_dirs)}] {case_id}", end=' ... ', flush=True)

        if not os.path.exists(nifti_out):
            ok = convert_dicom_series(dicom_dir, nifti_out)
            if not ok:
                print("SKIP")
                continue
            print("OK", end='')
        else:
            print("skip", end='')

        # GT: check for a Ground directory at the same level (CHAOS format)
        gt_path = ""
        ground_dir = os.path.join(os.path.dirname(dicom_dir), 'Ground')
        if os.path.isdir(ground_dir):
            if not os.path.exists(gt_out):
                ok = convert_ground_png(ground_dir, gt_out)
                if ok:
                    gt_path = gt_out
                    print(" + GT", end='')
            else:
                gt_path = gt_out
                print(" + GT(skip)", end='')
        print()

        modality, instruction = guess_modality_instruction(dicom_dir, case_id)

        rows.append({
            "case_id": case_id,
            "nifti_path": nifti_out,
            "gt_mask_path": gt_path,
            "instruction": instruction,
            "tool": "",
            "structure": "",
        })

    fields = ["case_id", "nifti_path", "gt_mask_path", "instruction", "tool", "structure"]
    with open(args.csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nToplam {len(rows)} case -> {args.csv_out}")

    if args.append_to and os.path.exists(args.append_to):
        with open(args.append_to, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writerows(rows)
        print(f"Eklendi: {args.append_to}")


if __name__ == '__main__':
    main()