#!/usr/bin/env python3
# =============================================================================
# validate_pipeline.py
#
# Agentic Segmentation Pipeline — Validation Script
#
# Her CSV satırı için pipeline'ı çalıştırır (VLM → Planner → Critic → Tool),
# GT maskı ile Dice skoru hesaplar ve sonuçları JSONL olarak kaydeder.
#
# CSV format (her satır bir case):
#   nifti_path, gt_mask_path, instruction, tool, structure, [gt_label_value]
#
# Çalıştırma:
#   python validate_pipeline.py --dataset_csv cases.csv --output_dir ./val_results
#
# Argümanlar:
#   --dataset_csv   : case listesi CSV dosyası (zorunlu)
#   --output_dir    : sonuçların yazılacağı dizin (default: ./val_results)
#   --max_cases     : çalıştırılacak maksimum case sayısı
#   --tool_filter   : sadece belirli tool'un case'lerini çalıştır
#   --force_tool    : agent'i bypass edip tüm case'lerde bu tool'u kullan
#
# Çıktılar:
#   output_dir/results.jsonl  — her satır bir case sonucu (JSON)
#   output_dir/summary.json   — toplam istatistikler
#   output_dir/masks/<case_id>/<tool>/  — üretilen mask dosyaları
#
# Servis URL'leri (env değişkenleri):
#   VLM_URL, LLM_URL, TOOL_URL_TOTALSEG, TOOL_URL_VOXTELL, TOOL_URL_BIOMEDPARSE
# =============================================================================

import argparse
import base64
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import requests
from scipy.ndimage import zoom

# Servis URL'leri — env'den al, yoksa localhost default'ları kullan
VLM_URL = os.environ.get("VLM_URL", "http://127.0.0.1:8001")
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8002")
TOOL_URL_TOTALSEG = os.environ.get("TOOL_URL_TOTALSEG", "http://127.0.0.1:8011")
TOOL_URL_VOXTELL = os.environ.get("TOOL_URL_VOXTELL", "http://127.0.0.1:8012")
TOOL_URL_BIOMEDPARSE = os.environ.get("TOOL_URL_BIOMEDPARSE", "http://127.0.0.1:8013")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/outputs")

TOOL_URLS = {
    "totalseg": TOOL_URL_TOTALSEG,
    "voxtell": TOOL_URL_VOXTELL,
    "biomedparse": TOOL_URL_BIOMEDPARSE,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_first_json(text: str) -> Optional[dict]:
    """Extract the first well-formed JSON object from an arbitrary string."""
    s = str(text)
    stack = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}" and stack > 0:
            stack -= 1
            if stack == 0 and start is not None:
                try:
                    d = json.loads(s[start:i+1])
                    if isinstance(d, dict):
                        return d
                except Exception:
                    pass
                start = None
    return None


def load_nifti(path: str) -> Optional[np.ndarray]:
    """Load a NIfTI or NPZ file and return the data array."""
    if not path or not os.path.exists(path):
        return None
    if os.path.basename(path).startswith("._"):
        return None
    if path.endswith(".npz"):
        npz = np.load(path, allow_pickle=True)
        key = next((k for k in ["mask", "seg", "segs", "pred", "arr_0"] if k in npz), list(npz.keys())[0])
        m = npz[key]
        return m[0] if m.ndim == 4 else m
    try:
        return nib.load(path).get_fdata()
    except Exception:
        return None


def find_best_pred_mask(mask_path: str, instruction: str) -> Optional[np.ndarray]:
    """Find the best matching mask file for the given instruction.

    If mask_path is a directory, scores each .nii.gz file by keyword overlap
    with the instruction and returns the best match (falls back to first file).
    If mask_path is a single file, loads it directly.
    """
    if not mask_path:
        return None
    if os.path.isfile(mask_path):
        return load_nifti(mask_path)
    if os.path.isdir(mask_path):
        nii_files = sorted([f for f in os.listdir(mask_path) if f.endswith(".nii.gz")])
        if not nii_files:
            return None
        instr_words = set(instruction.lower().replace("-", "_").replace(" ", "_").split("_"))
        instr_words = {w for w in instr_words if len(w) > 2}
        best_file = None
        best_score = -1
        for f in nii_files:
            fname = f.replace(".nii.gz", "").lower()
            score = sum(1 for w in instr_words if w in fname)
            if score > best_score:
                best_score = score
                best_file = f
        target = best_file if best_file else nii_files[0]
        return load_nifti(os.path.join(mask_path, target))
    return None


def find_best_gt_label(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Select the GT label with the highest overlap with the prediction.

    If there is no overlap at all, falls back to the highest label value
    (tumors/pathology are typically the last label in MSD datasets).
    """
    unique_labels = [l for l in np.unique(gt_mask).astype(int) if l > 0]
    if not unique_labels:
        return (gt_mask > 0).astype(np.float32)
    if len(unique_labels) == 1:
        return (gt_mask == unique_labels[0]).astype(np.float32)
    pred_bin = (pred_mask > 0)
    best_label = None
    best_overlap = -1
    for lbl in unique_labels:
        gt_bin = (gt_mask == lbl)
        overlap = int((pred_bin & gt_bin).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = lbl
    if best_overlap == 0:
        best_label = max(unique_labels)
    return (gt_mask == best_label).astype(np.float32)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = (pred > 0).astype(np.uint8)
    gt_b = (gt > 0).astype(np.uint8)
    inter = (pred_b & gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return float(2 * inter / denom)


def resize_to_match(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Resize pred to match gt's shape using nearest-neighbour zoom."""
    if pred.shape == gt.shape:
        return pred
    factors = [g / p for g, p in zip(gt.shape, pred.shape)]
    return zoom(pred, factors, order=0)


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------

def call_vlm(nifti_path: str, instruction: str) -> Dict[str, Any]:
    """Downsample the volume to 128³ and call the VLM /infer endpoint."""
    import tempfile
    img = nib.load(nifti_path)
    vol = img.get_fdata()
    if vol.ndim == 4:
        vol = vol[..., 0]
    factors = [128 / s for s in vol.shape[:3]]
    vol_small = zoom(vol, factors, order=1)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(nib.Nifti1Image(vol_small, np.eye(4)), tmp.name)
    payload = {
        "input_type": "3d",
        "nifti_b64": read_b64(tmp.name),
        "nii_axis": 2,
        "nii_num_slices": 16,
        "instruction": instruction,
        "max_new_tokens": 512,
        "temperature": 0.0,
        "use_think": False,
    }
    r = requests.post(f"{VLM_URL}/infer", json=payload, timeout=300)
    r.raise_for_status()
    os.unlink(tmp.name)
    return r.json()


# ---------------------------------------------------------------------------
# Tool call
# ---------------------------------------------------------------------------

def call_tool(tool: str, nifti_path: str, out_dir: str, params: dict) -> Dict[str, Any]:
    """Call a tool service's /infer endpoint.

    Handles 4-D (multi-channel) inputs by extracting the T1ce channel (index 1)
    before sending, since all tools expect 3-D volumes.
    """
    url = TOOL_URLS.get(tool)
    if not url:
        return {"ok": False, "message": f"Unknown tool: {tool}", "mask_path": None}
    os.makedirs(out_dir, exist_ok=True)
    actual_path = nifti_path
    try:
        import nibabel as _nib
        _img = _nib.load(nifti_path)
        if _img.ndim == 4:
            import numpy as _np
            _vol = _img.get_fdata()
            _ch = min(1, _vol.shape[3] - 1)  # prefer T1ce (index 1)
            _vol_3d = _vol[..., _ch]
            import tempfile as _tf
            _tmp = _tf.NamedTemporaryFile(suffix=".nii.gz", delete=False, dir=out_dir)
            _nib.save(_nib.Nifti1Image(_vol_3d.astype(_np.float32), _img.affine, _img.header), _tmp.name)
            actual_path = _tmp.name
    except Exception:
        pass
    payload = {"input_path": actual_path, "output_path": out_dir, "params": params}
    r = requests.post(f"{url}/infer", json=payload, timeout=600)
    r.raise_for_status()
    out = r.json()
    return {
        "ok": out.get("ok", out.get("status") == "success"),
        "message": out.get("message", ""),
        "mask_path": out.get("mask_path", out.get("output_path")),
    }


# ---------------------------------------------------------------------------
# Single case pipeline
# ---------------------------------------------------------------------------

def run_case(
    case_id: str,
    nifti_path: str,
    gt_mask_path: str,
    instruction: str,
    expected_tool: str,      # CSV'deki beklenen tool (sadece accuracy ölçümü için)
    structure: str,          # totalseg için yapı adı
    out_root: str,
    force_tool: str = None,  # agent'i bypass et, bu tool'u kullan
    gt_label_value: int = None,  # None → find_best_gt_label, int → sabit label
) -> Dict[str, Any]:
    """Run the full pipeline for one case and return a result dict."""
    result: Dict[str, Any] = {
        "case_id": case_id,
        "nifti_path": nifti_path,
        "instruction": instruction,
        "expected_tool": expected_tool,
        "selected_tool": None,
        "tool_ok": False,
        "dice": None,
        "timing": {},
        "error": None,
    }

    try:
        out_dir = os.path.join(out_root, case_id)
        os.makedirs(out_dir, exist_ok=True)

        # 1. VLM — force_tool modunda atlanır
        if force_tool:
            tool = force_tool
            result["selected_tool"] = tool
            result["critic_reason"] = f"force_tool={force_tool} (agent bypassed)"
            _inst_lower = instruction.lower()
            if any(w in _inst_lower for w in ["mri", "t1", "t2", "flair", "brain mri", "cardiac mri"]):
                _inferred_mod = "MRI"
            elif any(w in _inst_lower for w in ["ct", "ct scan", "abdominal ct"]):
                _inferred_mod = "CT"
            else:
                _inferred_mod = "CT"
            parsed = {"task_intent": {"target": structure or instruction}, "image": {"modality": {"label": _inferred_mod}}}
        else:
            pass

        vlm_resp = {}
        t0 = time.time()
        if not force_tool:
            vlm_resp = call_vlm(nifti_path, instruction)
            result["timing"]["vlm_s"] = round(time.time() - t0, 2)
            vlm_text = vlm_resp.get("text", "")
            parsed = extract_first_json(vlm_text)

        # 2. Tool seçimi — VLM → Planner → Critic üzerinden
        # expected_tool sadece accuracy ölçümü için saklanır, seçimde kullanılmaz
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'orchestrator'))
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), 'orchestrator'))
        try:
            if not force_tool:
                from clients import LLMClient as _LLMClient
                from agents import CriticAgent as _CriticAgent, PlannerAgent as _PlannerAgent
                from tool_registry import list_tools as _list_tools, tool_summary_for_llm as _tool_summary
                llm_client = _LLMClient(LLM_URL) if LLM_URL else None
                if not llm_client:
                    raise Exception("no LLM_URL")
                tools_compact = _tool_summary(_list_tools())
                planner_out = _PlannerAgent(llm_client).run(
                    parsed or {},
                    tools_compact=tools_compact,
                    user_instruction=instruction,
                )
                td, _ = _CriticAgent(llm_client).run(
                    planner_output=planner_out,
                    vlm_json=parsed or {},
                    tools_compact=tools_compact,
                    user_instruction=instruction,
                )
                if not td or not td.tool:
                    tool = planner_out.get("selected_tool", "voxtell")
                else:
                    tool = td.tool
                result["critic_reason"] = td.reason if td else ""
                result["planner_selected"] = planner_out.get("selected_tool")
        except Exception as ce:
            result["critic_error"] = str(ce)
            pass

        result["selected_tool"] = tool
        if expected_tool:
            result["tool_correct"] = (tool == expected_tool)

        # 3. Tool params
        params: Dict[str, Any] = {}
        if tool == "totalseg":
            if structure:
                params["structures"] = [s.strip() for s in structure.split(",")]
        elif tool == "voxtell":
            params["prompts"] = instruction
        elif tool == "biomedparse":
            params["prompts"] = instruction
            if parsed:
                mod = (parsed.get("image") or {}).get("modality", {})
                raw_mod = str((mod if isinstance(mod, dict) else {}).get("label", "CT")).upper()
                _MRI_ALIASES = {"T1", "T1-WEIGHTED", "T2", "T2-WEIGHTED", "FLAIR", "DWI", "MR", "MRI"}
                _CT_ALIASES = {"CECT", "NCCT", "CT"}
                if raw_mod in _MRI_ALIASES or "MRI" in raw_mod or "MR" in raw_mod:
                    params["modality"] = "MRI"
                elif raw_mod in _CT_ALIASES or "CT" in raw_mod:
                    params["modality"] = "CT"
                else:
                    params["modality"] = raw_mod
            else:
                if "mri" in instruction.lower() or "t1" in instruction.lower() or "brain" in instruction.lower() or "cardiac" in instruction.lower() or "hippocampus" in instruction.lower() or "atrium" in instruction.lower():
                    params["modality"] = "MRI"
                else:
                    params["modality"] = "CT"

        # 4. Tool çağır
        t1 = time.time()
        tool_out = call_tool(tool, nifti_path, os.path.join(out_dir, tool), params)
        result["timing"]["tool_s"] = round(time.time() - t1, 2)
        result["tool_ok"] = tool_out["ok"]

        if not tool_out["ok"]:
            result["error"] = tool_out.get("message")
            return result

        # 5. Instruction'a en uygun pred mask dosyasını bul
        mask_path = tool_out.get("mask_path")
        pred_mask = find_best_pred_mask(mask_path, instruction)

        if pred_mask is None:
            result["error"] = "Could not load prediction mask"
            return result

        # 6. GT mask yükle ve Dice hesapla
        if gt_mask_path and os.path.exists(gt_mask_path):
            gt_raw = load_nifti(gt_mask_path)
            if gt_raw is not None:
                if pred_mask.shape != gt_raw.shape:
                    pred_mask = resize_to_match(pred_mask, gt_raw)
                if gt_label_value is not None:
                    gt_mask = (gt_raw == int(gt_label_value)).astype(np.float32)
                    result["gt_label_used"] = int(gt_label_value)
                else:
                    gt_mask = find_best_gt_label(gt_raw, pred_mask)
                    result["gt_label_used"] = None
                result["dice"] = round(dice_score(pred_mask, gt_mask), 4)
                result["gt_unique_labels"] = [int(l) for l in np.unique(gt_raw) if l > 0]
        else:
            result["dice"] = None  # GT yok, sadece execution check

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["tb"] = traceback.format_exc()[:1000]

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate agentic segmentation pipeline")
    parser.add_argument("--dataset_csv", required=True, help="CSV with cases")
    parser.add_argument("--output_dir", default="./val_results", help="Where to save results")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--tool_filter", default=None, help="Only run cases for this tool (totalseg/voxtell/biomedparse)")
    parser.add_argument("--force_tool", default=None, help="Bypass agent, force this tool for ALL cases")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")

    cases = []
    with open(args.dataset_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)

    if args.tool_filter:
        cases = [c for c in cases if c.get("tool", "").strip() == args.tool_filter]
    if args.max_cases:
        cases = cases[:args.max_cases]

    print(f"Running validation on {len(cases)} cases...")
    print(f"Results -> {results_path}")
    print("-" * 60)

    all_results = []
    dice_by_tool: Dict[str, List[float]] = {"totalseg": [], "voxtell": [], "biomedparse": []}

    with open(results_path, "w") as fout:
        for i, row in enumerate(cases):
            case_id = row.get("case_id", f"case_{i:04d}")
            nifti_path = row.get("nifti_path", "").strip()
            gt_mask_path = row.get("gt_mask_path", "").strip()
            instruction = row.get("instruction", "Segment target structure").strip()
            tool = ""
            structure = row.get("structure", "").strip()
            _glv = row.get("gt_label_value", "").strip()
            gt_label_value = int(_glv) if _glv.isdigit() else None

            if not os.path.exists(nifti_path):
                print(f"[{i+1}/{len(cases)}] SKIP {case_id} — file not found: {nifti_path}")
                continue

            print(f"[{i+1}/{len(cases)}] {case_id} | tool={tool or 'auto'} | {instruction[:50]}", end=" ... ", flush=True)
            t_start = time.time()

            res = run_case(
                case_id=case_id,
                nifti_path=nifti_path,
                gt_mask_path=gt_mask_path,
                instruction=instruction,
                expected_tool=tool,
                structure=structure,
                out_root=os.path.join(args.output_dir, "masks"),
                force_tool=args.force_tool,
                gt_label_value=gt_label_value,
            )
            res["total_s"] = round(time.time() - t_start, 2)
            all_results.append(res)

            dice_str = f"Dice={res['dice']:.4f}" if res["dice"] is not None else "Dice=N/A"
            status = "OK" if res["tool_ok"] else f"FAIL({res.get('error', '')})"
            print(f"{status} | {dice_str} | {res['total_s']}s")

            sel_tool = res.get("selected_tool")
            if sel_tool and res["dice"] is not None:
                dice_by_tool.setdefault(sel_tool, []).append(res["dice"])

            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()

    total = len(all_results)
    ok_count = sum(1 for r in all_results if r["tool_ok"])
    all_dice = [r["dice"] for r in all_results if r["dice"] is not None]

    tool_correct_cases = [r for r in all_results if r.get("tool_correct") is not None]
    tool_accuracy = round(sum(1 for r in tool_correct_cases if r["tool_correct"]) / len(tool_correct_cases), 4) if tool_correct_cases else None

    summary = {
        "total_cases": total,
        "tool_ok": ok_count,
        "tool_fail": total - ok_count,
        "success_rate": round(ok_count / total, 4) if total else 0,
        "tool_selection_accuracy": tool_accuracy,
        "dice_overall": {
            "mean": round(float(np.mean(all_dice)), 4) if all_dice else None,
            "median": round(float(np.median(all_dice)), 4) if all_dice else None,
            "std": round(float(np.std(all_dice)), 4) if all_dice else None,
            "min": round(float(np.min(all_dice)), 4) if all_dice else None,
            "max": round(float(np.max(all_dice)), 4) if all_dice else None,
            "n": len(all_dice),
        },
        "dice_by_tool": {
            t: {
                "mean": round(float(np.mean(d)), 4) if d else None,
                "median": round(float(np.median(d)), 4) if d else None,
                "n": len(d),
            }
            for t, d in dice_by_tool.items() if d
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total cases  : {total}")
    print(f"Tool success : {ok_count}/{total} ({summary['success_rate']*100:.1f}%)")
    if all_dice:
        print(f"Dice overall : mean={summary['dice_overall']['mean']} | median={summary['dice_overall']['median']} | std={summary['dice_overall']['std']}")
    for t, d in summary["dice_by_tool"].items():
        print(f"  {t:15s}: mean={d['mean']} | median={d['median']} | n={d['n']}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()