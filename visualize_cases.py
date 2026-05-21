#!/usr/bin/env python3
# =============================================================================
# visualize_cases.py
#
# Segmentation Overlay Visualization
#
# CT slice üzerinde GT + 4 tool'un mask overlay'i.
# Her case için 1 satır, 6 sütun: [CT] [GT] [TotalSeg] [VoxTell] [BiomedParse] [Agentic]
#
# Çıktılar:
#   outputs/<case_name>.png      — her case için ayrı PNG
#   outputs/combined_overlay.png — 3 case'i üst üste gösteren birleşik figür
#
# Kullanım:
#   python visualize_cases.py
#
# Path'leri değiştirmek için VAL_DIR ve OUT_DIR sabitlerini düzenle.
# =============================================================================

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ── Path sabitleri — gerekirse değiştir ──────────────────────────────────────
VAL_DIR = "/scratch/project_2016517/furkan/val_data"
RES_DIR = VAL_DIR   # results_totalseg, results_voxtell, results_biomedparse, results burada
OUT_DIR = "/scratch/project_2016517/furkan/agentic-seg/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Renk paleti (tool başına sabit renk) ─────────────────────────────────────
C_GT = '#FFD700'   # altın — Ground Truth
C_TS = '#5C9EB8'   # çelik mavisi — TotalSegmentator
C_VX = '#E07A5F'   # terra cotta kırmızı — VoxTell
C_BM = '#F2CC8F'   # sıcak kum — BiomedParse
C_AG = '#81B29A'   # soft mint yeşil — Agentic Pipeline

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.bottom': False,
    'axes.spines.left': False,
})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_nifti(path):
    """Load a NIfTI file; return None if not found. Handles 4-D volumes."""
    if not os.path.exists(path):
        return None
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[:, :, :, 0]
    return data


def find_best_slice(mask_3d):
    """Return the axial slice index with the most non-zero pixels."""
    if mask_3d is None:
        return None
    sums = np.sum(mask_3d > 0, axis=(0, 1))
    if sums.max() == 0:
        return None
    return int(np.argmax(sums))


def get_ct_slice(ct_vol, slice_idx):
    """Extract and window an axial CT slice (abdominal window: center=40, width=400)."""
    sl = ct_vol[:, :, slice_idx].T
    wl, ww = 40, 400
    vmin, vmax = wl - ww / 2, wl + ww / 2
    sl = np.clip(sl, vmin, vmax)
    sl = (sl - vmin) / (vmax - vmin)
    return sl


def get_mask_slice(mask_vol, slice_idx, label_value=None):
    """Extract a binary mask slice; optionally filter by label_value."""
    if mask_vol is None:
        return None
    sl = mask_vol[:, :, slice_idx].T
    if label_value is not None:
        sl = (sl == label_value).astype(np.float32)
    else:
        sl = (sl > 0).astype(np.float32)
    return sl


def overlay_mask(ax, ct_slice, mask_slice, color_hex, alpha=0.4, title="", dice=None):
    """Render a coloured mask overlay + contour on top of a CT slice."""
    ax.imshow(ct_slice, cmap='gray', aspect='equal')
    if mask_slice is not None and mask_slice.max() > 0:
        rgb = mcolors.to_rgb(color_hex)
        colored = np.zeros((*mask_slice.shape, 4))
        colored[mask_slice > 0] = [rgb[0], rgb[1], rgb[2], alpha]
        ax.imshow(colored, aspect='equal')
        ax.contour(mask_slice, levels=[0.5], colors=[color_hex], linewidths=1.5)
    title_text = title
    if dice is not None:
        title_text += f'\nDice={dice:.3f}'
    ax.set_title(title_text, fontsize=20, fontweight='bold', color=color_hex)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

CASES = [
    {
        "name": "liver_tumor_case",
        "title": "Liver Tumor — vx_liver_tumor_060",
        "case_id": "vx_liver_tumor_060",
        "ct_path": f"{VAL_DIR}/Task03_Liver/imagesTr/liver_43.nii.gz",
        "gt_path": f"{VAL_DIR}/Task03_Liver/labelsTr/liver_43.nii.gz",
        "gt_label": 2,
        "masks": {
            "TotalSeg":    None,
            "VoxTell":     f"{RES_DIR}/results_voxtell/masks/vx_liver_tumor_060/voxtell/liver_43.nii.gz",
            "BiomedParse": f"{RES_DIR}/results_biomedparse/masks/vx_liver_tumor_060/biomedparse/liver_43_seg.nii.gz",
            "Agentic":     f"{RES_DIR}/results/masks/vx_liver_tumor_060/voxtell/liver_43.nii.gz",
        },
        "dice": {"TotalSeg": 0.000, "VoxTell": 0.921, "BiomedParse": 0.005, "Agentic": 0.921},
    },
    {
        "name": "liver_organ_case",
        "title": "Liver Organ — ts_liver_032",
        "case_id": "ts_liver_032",
        "ct_path": f"{VAL_DIR}/Task03_Liver/imagesTr/liver_127.nii.gz",
        "gt_path": f"{VAL_DIR}/Task03_Liver/labelsTr/liver_127.nii.gz",
        "gt_label": 1,
        "masks": {
            "TotalSeg":    f"{RES_DIR}/results_totalseg/masks/ts_liver_032/totalseg/liver.nii.gz",
            "VoxTell":     f"{RES_DIR}/results_voxtell/masks/ts_liver_032/voxtell/liver_127.nii.gz",
            "BiomedParse": f"{RES_DIR}/results_biomedparse/masks/ts_liver_032/biomedparse/liver_127_seg.nii.gz",
            "Agentic":     f"{RES_DIR}/results/masks/ts_liver_032/totalseg/liver.nii.gz",
        },
        "dice": {"TotalSeg": 0.976, "VoxTell": 0.025, "BiomedParse": 0.972, "Agentic": 0.976},
    },
    {
        "name": "lung_tumor_case",
        "title": "Lung Tumor — vx_lung_tumor_043",
        "case_id": "vx_lung_tumor_043",
        "ct_path": f"{VAL_DIR}/Task06_Lung/imagesTr/lung_064.nii.gz",
        "gt_path": f"{VAL_DIR}/Task06_Lung/labelsTr/lung_064.nii.gz",
        "gt_label": 1,
        "masks": {
            "TotalSeg":    None,
            "VoxTell":     f"{RES_DIR}/results_voxtell/masks/vx_lung_tumor_043/voxtell/lung_064.nii.gz",
            "BiomedParse": f"{RES_DIR}/results_biomedparse/masks/vx_lung_tumor_043/biomedparse/lung_064_seg.nii.gz",
            "Agentic":     f"{RES_DIR}/results/masks/vx_lung_tumor_043/voxtell/lung_064.nii.gz",
        },
        "dice": {"TotalSeg": 0.000, "VoxTell": 0.940, "BiomedParse": 0.008, "Agentic": 0.940},
    },
]

TOOL_COLORS = {"GT": C_GT, "TotalSeg": C_TS, "VoxTell": C_VX, "BiomedParse": C_BM, "Agentic": C_AG}
TOOL_ORDER = ["TotalSeg", "VoxTell", "BiomedParse", "Agentic"]


# ---------------------------------------------------------------------------
# Per-case figures
# ---------------------------------------------------------------------------

for case in CASES:
    print(f"\n{'='*60}")
    print(f"  {case['title']}")
    print(f"{'='*60}")

    ct_vol = load_nifti(case["ct_path"])
    gt_vol = load_nifti(case["gt_path"])

    if ct_vol is None:
        print(f"  ⚠ CT bulunamadı: {case['ct_path']}")
        continue
    if gt_vol is None:
        print(f"  ⚠ GT bulunamadı: {case['gt_path']}")
        continue

    gt_binary = (gt_vol == case["gt_label"]).astype(np.float32)
    best_slice = find_best_slice(gt_binary)
    if best_slice is None:
        print(f"  ⚠ GT'de label={case['gt_label']} bulunamadı")
        continue

    print(f"  CT shape: {ct_vol.shape}")
    print(f"  Best slice: {best_slice}")

    pred_masks = {}
    for tool in TOOL_ORDER:
        mask_path = case["masks"].get(tool)
        if mask_path and os.path.exists(mask_path):
            pred_masks[tool] = load_nifti(mask_path)
            print(f"  {tool}: loaded {mask_path}")
        else:
            pred_masks[tool] = None
            reason = "unsupported" if mask_path is None else "not found"
            print(f"  {tool}: {reason}")

    ct_slice = get_ct_slice(ct_vol, best_slice)
    gt_slice = get_mask_slice(gt_vol, best_slice, label_value=case["gt_label"])

    fig, axes = plt.subplots(1, 6, figsize=(20, 3.5))

    axes[0].imshow(ct_slice, cmap='gray', aspect='equal')
    axes[0].set_title('CT Input', fontsize=20, fontweight='bold')
    axes[0].set_xticks([]); axes[0].set_yticks([])

    overlay_mask(axes[1], ct_slice, gt_slice, C_GT, alpha=0.35, title='Ground Truth')

    for i, tool in enumerate(TOOL_ORDER):
        ax = axes[i + 2]
        mask_slice = get_mask_slice(pred_masks[tool], best_slice) if pred_masks[tool] is not None else None
        dice_val = case["dice"].get(tool, 0.0)
        overlay_mask(ax, ct_slice, mask_slice, TOOL_COLORS[tool], alpha=0.4, title=tool, dice=dice_val)
        if gt_slice is not None and gt_slice.max() > 0:
            ax.contour(gt_slice, levels=[0.5], colors=['#FFD700'], linewidths=0.8, linestyles='dashed')

    plt.suptitle(case["title"], fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout(pad=0.3)

    save_path = f"{OUT_DIR}/{case['name']}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  >> Saved: {save_path}")


# ---------------------------------------------------------------------------
# Combined figure — all 3 cases stacked
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  Combined figure (3 rows)")
print(f"{'='*60}")

fig, axes = plt.subplots(3, 6, figsize=(20, 10.5))

for row, case in enumerate(CASES):
    ct_vol = load_nifti(case["ct_path"])
    gt_vol = load_nifti(case["gt_path"])
    if ct_vol is None or gt_vol is None:
        continue

    gt_binary = (gt_vol == case["gt_label"]).astype(np.float32)
    best_slice = find_best_slice(gt_binary)
    if best_slice is None:
        continue

    ct_slice = get_ct_slice(ct_vol, best_slice)
    gt_slice = get_mask_slice(gt_vol, best_slice, label_value=case["gt_label"])

    pred_masks = {}
    for tool in TOOL_ORDER:
        mask_path = case["masks"].get(tool)
        pred_masks[tool] = load_nifti(mask_path) if mask_path and os.path.exists(mask_path) else None

    axes[row][0].imshow(ct_slice, cmap='gray', aspect='equal')
    if row == 0:
        axes[row][0].set_title('CT Input', fontsize=20, fontweight='bold', color='white')
    axes[row][0].set_ylabel(case["title"].split("—")[0].strip(),
                            fontsize=24, fontweight='bold', color='white', rotation=90)
    axes[row][0].set_xticks([]); axes[row][0].set_yticks([])

    overlay_mask(axes[row][1], ct_slice, gt_slice, C_GT, alpha=0.35, title='Ground Truth')

    for i, tool in enumerate(TOOL_ORDER):
        ax = axes[row][i + 2]
        mask_slice = get_mask_slice(pred_masks[tool], best_slice) if pred_masks[tool] is not None else None
        dice_val = case["dice"].get(tool, 0.0)
        title = tool if row == 0 else ''
        overlay_mask(ax, ct_slice, mask_slice, TOOL_COLORS[tool], alpha=0.4, title=title, dice=dice_val)
        if gt_slice is not None and gt_slice.max() > 0:
            ax.contour(gt_slice, levels=[0.5], colors=['#FFD700'], linewidths=0.8, linestyles='dashed')

fig.patch.set_facecolor('black')
plt.tight_layout(pad=0.3)
save_path = f"{OUT_DIR}/combined_overlay.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
plt.close()
print(f"  >> Saved: {save_path}")

print("\n  ✅ Visualization tamamlandı!")