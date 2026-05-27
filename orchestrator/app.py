# =============================================================================
# orchestrator/app.py
#
# Gradio web UI and main orchestrator logic for the Agentic Segmentation system.
#
# Pipeline (triggered by the "Run" button)
# -----------------------------------------
#   1. VLM       — call_hulu_infer() sends the NIfTI volume to the Hulu VLM
#                  service, which returns a structured JSON analysis of the
#                  image (modality, anatomy, task intent).
#   2. Planner   — PlannerAgent reads the VLM output + tool registry and
#                  proposes a trajectory (tool + target).
#   3. Critic    — CriticAgent validates the Planner's decision via LLM.
#   4. Explainer — ExplainerAgent generates a clarification message when the
#                  Critic raises need_clarification.
#   5. Executor  — POSTs to the selected tool service (TotalSeg / VoxTell /
#                  BiomedParse) and loads the resulting mask(s) into state.
#   6. Verifier  — VerifierAgent.run() checks mask validity deterministically.
#   7. Reporter  — ReporterAgent writes a concise final summary.
#
# Environment variables
# ---------------------
#   VLM_URL               — Hulu VLM service base URL
#   LLM_URL               — Qwen LLM service base URL (optional; enables agents)
#   TOOL_URL_TOTALSEG     — TotalSegmentator service base URL
#   TOOL_URL_VOXTELL      — VoxTell service base URL
#   TOOL_URL_BIOMEDPARSE  — BiomedParse service base URL
#   OUTPUT_DIR            — directory for plan/trace JSON artefacts
#   PORT                  — Gradio server port (default: 7860)
# =============================================================================

from __future__ import annotations

import requests
import base64
import os, time, json
import traceback
import shutil
from typing import Optional, Tuple, Dict, Any, List
import gradio as gr
import numpy as np
import nibabel as nib

from contracts import (now_run_id, save_json)
from clients import VLMClient, LLMClient
from agents import CriticAgent, ExplainerAgent, ReporterAgent, PlannerAgent, VerifierAgent
from tool_registry import list_tools, tool_summary_for_llm


# ---------------------------------------------------------------------------
# Tool-specific overlay colours for the mask viewer
# ---------------------------------------------------------------------------
TOOL_COLORS = {
    "totalseg":    (92, 158, 184),   # teal / cyan
    "voxtell":     (224, 122, 95),   # warm orange
    "biomedparse": (242, 204, 143),  # sand / gold
}
FALLBACK_COLOR = (129, 178, 154)     # muted green

# Default tool service URLs (overridden by TOOL_URL_* env vars)
DEFAULT_TOOL_URLS = {
    "totalseg":    "http://127.0.0.1:8011",
    "voxtell":     "http://127.0.0.1:8012",
    "biomedparse": "http://127.0.0.1:8013",
}


# ---------------------------------------------------------------------------
# JSON serialisation helper — never returns empty, never raises
# ---------------------------------------------------------------------------

def _safe_json_dumps(obj: Any, label: str = "unknown") -> str:
    """Serialize *obj* to a JSON string.

    Three-tier fallback:
      1. json.dumps with default=str   (handles numpy int64, datetime, etc.)
      2. Replace non-serialisable values with a type placeholder
      3. Return a minimal error dict with repr()
    """
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception as e1:
        try:
            cleaned = json.loads(
                json.dumps(obj, default=lambda x: f"<{type(x).__name__}>")
            )
            return json.dumps(cleaned, indent=2)
        except Exception as e2:
            return json.dumps({
                "_serialization_error": f"Could not serialize {label}: {e1}",
                "_fallback_error": str(e2),
                "_repr": repr(obj)[:3000],
            }, indent=2)


# ---------------------------------------------------------------------------
# Volume / image utilities
# ---------------------------------------------------------------------------

def _normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """Clip to [1st, 99th] percentile and scale to uint8 [0, 255]."""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    x = (np.clip(x, lo, hi) - lo) / (hi - lo)
    return (x * 255.0).astype(np.uint8)


def _slice_from_vol(vol: np.ndarray, axis: int, z: int) -> np.ndarray:
    """Extract a 2-D slice at index *z* along *axis* (0=sag, 1=cor, 2=ax)."""
    if axis == 0:
        return vol[int(np.clip(z, 0, vol.shape[0]-1)), :, :]
    elif axis == 1:
        return vol[:, int(np.clip(z, 0, vol.shape[1]-1)), :]
    else:
        return vol[:, :, int(np.clip(z, 0, vol.shape[2]-1))]


def _max_index(vol: np.ndarray, axis: int) -> int:
    """Return the last valid slice index along *axis*."""
    return int(vol.shape[axis] - 1)


def _read_file_b64(path: str) -> str:
    """Read a binary file and return its base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_first_json_obj(text: str) -> Optional[dict]:
    """Extract the first well-formed JSON object from an arbitrary string.

    Falls back to a single-quote → double-quote replacement if the initial
    parse fails.  Returns None if no valid object is found.
    """
    if text is None:
        return None
    s = str(text)
    stack = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    cand = s[start:i+1]
                    try:
                        d = json.loads(cand)
                        if isinstance(d, dict):
                            return d
                    except Exception:
                        pass
                    try:
                        d = json.loads(cand.replace("'", '"'))
                        if isinstance(d, dict):
                            return d
                    except Exception:
                        pass
                    start = None
    return None


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------

def call_hulu_infer(nifti_path: str, axis: int, num_slices: int, instruction: str) -> dict:
    """Downsample the NIfTI volume to 64³, then call the Hulu VLM /infer endpoint.

    Downsampling keeps the payload small enough for the VLM to handle quickly
    without losing the structural information needed for modality / anatomy
    detection.  Returns the raw response dict from the VLM service.
    """
    import tempfile
    from scipy.ndimage import zoom
    img = nib.load(nifti_path)
    vol = img.get_fdata()
    if vol.ndim == 4:
        vol = vol[..., 0]
    factors = [64 / s for s in vol.shape[:3]]
    vol_small = zoom(vol, factors, order=1)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(nib.Nifti1Image(vol_small, np.eye(4)), tmp.name)
    base = os.environ.get("VLM_URL", "http://127.0.0.1:8001").rstrip("/")
    payload = {
        "input_type": "3d",
        "nifti_b64": _read_file_b64(tmp.name),
        "nii_axis": int(axis),
        "nii_num_slices": int(min(num_slices, 16)),
        "instruction": str(instruction),
        "max_new_tokens": 512,
        "temperature": 0.0,
        "use_think": False,
    }
    r = requests.post(f"{base}/infer", json=payload, timeout=300)
    r.raise_for_status()
    os.unlink(tmp.name)
    return r.json()


# ---------------------------------------------------------------------------
# Overlay helper — composites a coloured mask onto a greyscale CT slice
# ---------------------------------------------------------------------------

def _make_overlay(ct_slice: np.ndarray, mask_slice: np.ndarray,
                  tool_name: str = "") -> np.ndarray:
    """Blend a segmentation mask onto a CT slice using the tool's colour."""
    ct_rgb = np.stack([_normalize_to_uint8(ct_slice)] * 3, axis=-1)
    mask_bin = (mask_slice > 0).astype(np.uint8)
    r, g, b = TOOL_COLORS.get(tool_name.lower(), FALLBACK_COLOR)
    overlay = ct_rgb.copy()
    alpha = 0.55
    overlay[mask_bin == 1, 0] = np.clip(
        ct_rgb[mask_bin == 1, 0] * (1 - alpha) + r * alpha, 0, 255).astype(np.uint8)
    overlay[mask_bin == 1, 1] = np.clip(
        ct_rgb[mask_bin == 1, 1] * (1 - alpha) + g * alpha, 0, 255).astype(np.uint8)
    overlay[mask_bin == 1, 2] = np.clip(
        ct_rgb[mask_bin == 1, 2] * (1 - alpha) + b * alpha, 0, 255).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Gradio event handlers — volume loading and preview
# ---------------------------------------------------------------------------

def on_load_nifti(nifti_file):
    """Load a NIfTI file into session state and return an initial preview slice."""
    if nifti_file is None:
        return {}, None, gr.update(), gr.update(), "No file.", gr.update(choices=[], value=None)
    path = nifti_file.name if hasattr(nifti_file, "name") else str(nifti_file)
    img = nib.load(path)
    vol = np.asarray(img.get_fdata())
    if vol.ndim == 4:
        vol = vol[..., 0]
    axis = 2
    zmax = _max_index(vol, axis)
    z0 = zmax // 2
    preview = _normalize_to_uint8(_slice_from_vol(vol, axis, z0))
    st = {"path": path, "shape": list(vol.shape), "vol": vol,
          "mask_dir": None, "masks": {}, "tool_name": ""}
    z_slider = gr.update(minimum=0, maximum=zmax, value=z0, step=1, interactive=True)
    return (st, preview, z_slider, gr.update(value=axis),
            f"Loaded: {os.path.basename(path)} | shape={vol.shape}",
            gr.update(choices=[], value=None))


def on_preview_change(state, axis, z):
    """Re-render the preview slice when axis or z-slider changes."""
    if not state or "vol" not in state:
        return None, gr.update()
    vol = state["vol"]
    zmax = _max_index(vol, axis)
    z = int(np.clip(z, 0, zmax))
    return _normalize_to_uint8(_slice_from_vol(vol, axis, z)), gr.update(maximum=zmax, value=z)


def on_mask_select(state, axis, z, selected_mask):
    """Overlay the selected mask on the current CT slice."""
    if not state or "vol" not in state or not selected_mask:
        return None
    vol = state["vol"]
    masks = state.get("masks", {})
    tool_name = state.get("tool_name", "")
    zmax = _max_index(vol, axis)
    z = int(np.clip(z, 0, zmax))
    ct_slice = _slice_from_vol(vol, axis, z)
    if selected_mask in masks:
        mask_vol = masks[selected_mask]
        if mask_vol.shape == vol.shape:
            mask_slice = _slice_from_vol(mask_vol, axis, z)
            return _make_overlay(ct_slice, mask_slice, tool_name)
    return _normalize_to_uint8(ct_slice)


def on_z_or_axis_change(state, axis, z, mask_dropdown):
    """Refresh both the preview slice and the mask overlay when axis/z changes."""
    preview, z_update = on_preview_change(state, axis, z)
    overlay_img = on_mask_select(state, axis, z, mask_dropdown)
    return preview, z_update, overlay_img


# ---------------------------------------------------------------------------
# Report builder — produces the text shown in the "Agent Report" box
# ---------------------------------------------------------------------------

def _build_report(plan, decision, trace, resp, vlm_text, parsed_found,
                  explainer_text, reporter_text, tool_result, mask_dropdown_choices):
    """Build a formatted report with a user-friendly summary paragraph.

    The report has two parts:
      • Structured agent details (VLM, Planner, Critic, Executor, etc.)
      • A plain-English SUMMARY paragraph at the end for non-technical readers
    """
    if not parsed_found:
        return (
            f"VLM OK | model={resp.get('model_id')} | device={resp.get('device')}\n"
            f"VLM JSON parse FAILED\n\n{vlm_text[:1500]}"
        )

    _mod = plan.get("image", {}).get("modality", {}) or {}
    _anat = plan.get("image", {}).get("anatomy", {}) or {}
    _task = plan.get("task_intent", {}) or {}
    _amb = plan.get("ambiguity", {}) or {}

    lines = []
    lines.append("=" * 70)
    lines.append("AGENTIC SEGMENTATION PIPELINE — FINAL REPORT")
    lines.append("=" * 70)

    # -- User request -------------------------------------------------------
    lines.append(f"\n📋 USER REQUEST")
    lines.append(f"   Raw instruction : {plan.get('user_raw', '')}")
    lines.append(f"   Intent summary  : {plan.get('intent_summary', '')}")
    if plan.get("message_to_user"):
        lines.append(f"   Message to user : {plan.get('message_to_user')}")

    # -- VLM image analysis -------------------------------------------------
    lines.append(f"\n🖼️  IMAGE ANALYSIS (VLM)")
    lines.append(f"   Model           : {resp.get('model_id')}")
    lines.append(f"   Device          : {resp.get('device')}")
    lines.append(f"   Modality        : {str(_mod.get('label','unknown')).upper()} (confidence: {_mod.get('confidence','?')})")
    lines.append(f"   Anatomy         : {_anat.get('region','unknown')} (confidence: {_anat.get('confidence','?')})")
    if _mod.get("notes"):
        lines.append(f"   Modality notes  : {_mod.get('notes')}")
    if _anat.get("notes"):
        lines.append(f"   Anatomy notes   : {_anat.get('notes')}")

    # -- Task ---------------------------------------------------------------
    lines.append(f"\n🎯 TASK")
    lines.append(f"   Type            : {_task.get('type','unknown')}")
    lines.append(f"   Target          : {_task.get('target','unknown')}")

    # -- Evidence -----------------------------------------------------------
    evidence = plan.get("evidence", []) or []
    if evidence:
        lines.append(f"\n🔍 EVIDENCE (from VLM)")
        for e in evidence:
            lines.append(f"   • {e}")

    # -- Ambiguity ----------------------------------------------------------
    if _amb and _amb.get("needs_clarification") is not None:
        lines.append(f"\n⚠️  AMBIGUITY")
        lines.append(f"   Needs clarification : {_amb.get('needs_clarification')}")
        reasons = _amb.get("reasons", [])
        if isinstance(reasons, list):
            for r in reasons:
                lines.append(f"   • {r}")
        elif reasons:
            lines.append(f"   • {reasons}")

    # -- Planner trajectory -------------------------------------------------
    traj = plan.get("trajectory", []) or []
    if traj:
        lines.append(f"\n🗺️  PLANNER TRAJECTORY")
        for s in traj:
            step = s.get('step', '?')
            action = s.get('action', '?')
            detail = s.get('detail', '')
            lines.append(f"   {step}. [{action}] {detail}")

    # -- Critic decision ----------------------------------------------------
    lines.append(f"\n🧠 CRITIC DECISION")
    lines.append(f"   Source          : {trace.get('critic',{}).get('source','unknown')}")
    lines.append(f"   Action          : {decision.get('action','unknown')}")
    lines.append(f"   Selected tool   : {decision.get('tool') or '(none)'}")
    lines.append(f"   Reason          : {decision.get('reason','')}")

    # -- Clarification questions --------------------------------------------
    questions = plan.get("questions", []) or []
    if decision.get("action") == "need_clarification" or questions:
        lines.append(f"\n❓ CLARIFICATION NEEDED")
        for q in questions:
            lines.append(f"   ? {q}")
        if explainer_text:
            lines.append(f"\n💬 EXPLAINER")
            lines.append(f"   {explainer_text}")

    # -- Executor result ----------------------------------------------------
    if tool_result:
        verif = plan.get("verification", {})
        verif_str = "✅ OK" if verif.get("ok") else f"❌ FAILED: {verif.get('issues', [])}"
        lines.append(f"\n⚙️  EXECUTOR")
        lines.append(f"   Tool            : {tool_result.get('tool')}")
        lines.append(f"   Status          : {'✅ OK' if tool_result.get('ok') else '❌ FAILED'}")
        lines.append(f"   Message         : {tool_result.get('message')}")
        lines.append(f"   Mask path       : {tool_result.get('mask_path')}")
        lines.append(f"   Verification    : {verif_str}")
        if mask_dropdown_choices:
            lines.append(f"   Available masks : {', '.join(mask_dropdown_choices)}")

    # -- Reporter text (from LLM) ------------------------------------------
    if reporter_text:
        lines.append(f"\n📝 REPORTER SUMMARY")
        lines.append(f"   {reporter_text}")

    # ======================================================================
    # USER-FRIENDLY SUMMARY PARAGRAPH
    # ======================================================================
    lines.append(f"\n" + "=" * 70)
    lines.append("📄 SUMMARY")
    lines.append("=" * 70)

    mod_label = str(_mod.get('label', 'unknown')).upper()
    anat_region = str(_anat.get('region', 'unknown'))
    task_type = str(_task.get('type', 'segmentation'))
    task_target = str(_task.get('target', 'unknown'))
    tool_name = decision.get('tool') or 'none'
    user_raw = plan.get('user_raw', '')

    summary_parts = []

    summary_parts.append(
        f'The user requested: "{user_raw}". '
        f'The VLM analyzed the uploaded volume and identified it as a {mod_label} scan '
        f'of the {anat_region} region.'
    )

    if evidence:
        evidence_str = "; ".join(str(e) for e in evidence[:3])
        summary_parts.append(
            f'Key observations from the image analysis include: {evidence_str}.'
        )

    summary_parts.append(
        f'Based on this, the Planner determined the task as "{task_type}" '
        f'targeting "{task_target}", and the Critic selected "{tool_name}" as the '
        f'most appropriate segmentation tool.'
    )

    if decision.get("action") == "need_clarification":
        q_str = " ".join(str(q) for q in questions[:3]) if questions else "additional details"
        summary_parts.append(
            f'However, the system flagged that clarification may be needed: {q_str}'
        )
    elif tool_result:
        if tool_result.get("ok"):
            mask_count = len(mask_dropdown_choices)
            if mask_count > 0:
                mask_names = ", ".join(mask_dropdown_choices[:5])
                extra = f" (and {mask_count - 5} more)" if mask_count > 5 else ""
                summary_parts.append(
                    f'The tool executed successfully and produced {mask_count} '
                    f'segmentation mask(s): {mask_names}{extra}. '
                    f'You can select and overlay them on the CT preview using the '
                    f'dropdown above.'
                )
            else:
                summary_parts.append(
                    f'The tool executed successfully. The output mask is available '
                    f'at: {tool_result.get("mask_path", "N/A")}.'
                )
            verif = plan.get("verification", {})
            if verif.get("ok"):
                summary_parts.append(
                    'The Verifier confirmed that the output is anatomically '
                    'consistent and structurally valid.'
                )
            elif verif.get("issues"):
                issues_str = ", ".join(str(i) for i in verif["issues"])
                summary_parts.append(
                    f'Note: the Verifier raised some concerns: {issues_str}. '
                    f'Please review the overlay carefully.'
                )
        else:
            summary_parts.append(
                f'Unfortunately, the tool execution failed with the following '
                f'message: {tool_result.get("message", "unknown error")}.'
            )
    else:
        if decision.get("action") == "error":
            summary_parts.append(
                f'The pipeline could not proceed to execution. '
                f'Reason: {decision.get("reason", "unknown")}.'
            )

    if reporter_text:
        summary_parts.append(f'Reporter note: {reporter_text}')

    lines.append("\n" + " ".join(summary_parts))
    lines.append(f"\n" + "=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline handler
# ---------------------------------------------------------------------------

def on_run(state, axis, z, mode, n_slices, instruction, user_override=False):
    """Main pipeline: VLM → Planner → Critic → Executor → Verifier → Reporter.

    Parameters
    ----------
    state         : Gradio session state dict (contains loaded volume + masks)
    axis          : viewing axis (0/1/2)
    z             : current slice index
    mode          : "3D volume" | "Multi-slice" | "2D slice"
    n_slices      : number of slices to send to the VLM
    instruction   : free-text segmentation goal from the user
    user_override : True when the user clicked "Proceed anyway" after a
                    need_clarification response
    """
    # -- Guard: no volume loaded --------------------------------------------
    if not state or "vol" not in state:
        _e = {"status": "error", "reason": "No NIfTI loaded. Please load a volume first."}
        return (state, None, None,
                "Error: Load NIfTI first.",
                _safe_json_dumps(_e, "plan"), _safe_json_dumps(_e, "trace"),
                gr.update(choices=[], value=None))

    # -- Guard: empty instruction -------------------------------------------
    if not instruction or not str(instruction).strip():
        preview, _ = on_preview_change(state, axis, z)
        _e = {"status": "error", "reason": "Instruction field is empty."}
        return (state, preview, None,
                "Error: instruction empty.",
                _safe_json_dumps(_e, "plan"), _safe_json_dumps(_e, "trace"),
                gr.update(choices=[], value=None))

    vol = state["vol"]
    run_id = now_run_id()
    t0 = time.time()
    zmax = _max_index(vol, axis)
    z = int(np.clip(z, 0, zmax))
    img2d_u8 = _normalize_to_uint8(_slice_from_vol(vol, axis, z))

    # -- Guard: unsupported mode --------------------------------------------
    if mode != "3D volume":
        _e = {"status": "error", "reason": f"Mode '{mode}' is not supported. Only '3D volume' is enabled."}
        return (state, img2d_u8, None,
                "Only '3D volume' mode is enabled.",
                _safe_json_dumps(_e, "plan"), _safe_json_dumps(_e, "trace"),
                gr.update(choices=[], value=None))

    # -- Guard: file path ---------------------------------------------------
    nifti_path = state.get("path", "")
    if not nifti_path or not os.path.exists(nifti_path):
        _e = {"status": "error", "reason": f"NIfTI path not found: {nifti_path}"}
        return (state, img2d_u8, None,
                "NIfTI path not found.",
                _safe_json_dumps(_e, "plan"), _safe_json_dumps(_e, "trace"),
                gr.update(choices=[], value=None))

    # -- Initialise trace dict ----------------------------------------------
    trace = {
        "run_id": run_id, "events": [], "timing": {},
        "inputs": {
            "path": nifti_path, "axis": int(axis), "z": int(z),
            "mode": mode, "n_slices": int(n_slices),
            "instruction": str(instruction).strip(),
        },
        "vlm": {}, "planner": {}, "critic": {}, "explainer": {},
        "executor": {}, "verifier": {}, "reporter": {},
    }

    def log(msg, **extra):
        e = {"t": time.time(), "msg": msg}
        if extra:
            e["extra"] = extra
        trace["events"].append(e)

    log("Run clicked")

    try:
        # ==================================================================
        # 1. VLM — image understanding (analyse image only, no user goal)
        # ==================================================================
        t_vlm0 = time.time()
        vlm_prompt = (
            "Analyze this medical image. Describe modality, anatomy, "
            "and any findings visible in the image only."
        )
        resp = call_hulu_infer(nifti_path, axis, n_slices, vlm_prompt)
        trace["timing"]["vlm_s"] = float(time.time() - t_vlm0)
        vlm_text = str(resp.get("text", ""))
        trace["vlm"]["response_meta"] = {
            "model_id": resp.get("model_id"),
            "device": resp.get("device"),
        }
        trace["vlm"]["text_head"] = vlm_text[:2000]
        parsed = _extract_first_json_obj(vlm_text)
        parsed_found = isinstance(parsed, dict)
        trace["vlm"]["parsed_json_found"] = parsed_found

        # ==================================================================
        # 2. Plan build — normalise VLM output into a canonical plan dict
        # ==================================================================
        if parsed_found:
            user_blk = parsed.get("user", {}) if isinstance(parsed.get("user", {}), dict) else {}
            img_blk = parsed.get("image", {}) if isinstance(parsed.get("image", {}), dict) else {}
            mod_blk = img_blk.get("modality", {}) if isinstance(img_blk.get("modality", {}), dict) else {}
            anat_blk = img_blk.get("anatomy", {}) if isinstance(img_blk.get("anatomy", {}), dict) else {}
            task = parsed.get("task_intent", {}) if isinstance(parsed.get("task_intent", {}), dict) else {}

            raw_evidence = parsed.get("evidence", [])
            if isinstance(raw_evidence, dict):
                evidence = raw_evidence.get("summary", [])
            elif isinstance(raw_evidence, list):
                evidence = raw_evidence
            else:
                evidence = []
            if isinstance(evidence, str):
                evidence = [evidence]
            if not isinstance(evidence, list):
                evidence = []

            raw_questions = parsed.get("questions", [])
            if isinstance(raw_questions, str):
                questions = [raw_questions]
            elif isinstance(raw_questions, list):
                questions = raw_questions
            else:
                questions = []

            plan = {
                "run_id": run_id,
                "goal": instruction,
                "status": parsed.get("status", "ok"),
                "user_raw": instruction,
                "intent_summary": user_blk.get("intent_summary", ""),
                "ambiguity": user_blk.get("ambiguity", {}) if isinstance(user_blk.get("ambiguity", {}), dict) else {},
                "image": {
                    "modality": {
                        "label": mod_blk.get("label", "unknown"),
                        "confidence": mod_blk.get("confidence", None),
                        "notes": mod_blk.get("notes", ""),
                    },
                    "anatomy": {
                        "region": anat_blk.get("region", "unknown"),
                        "confidence": anat_blk.get("confidence", None),
                        "notes": anat_blk.get("notes", ""),
                    },
                },
                "task_intent": task,
                "evidence": evidence,
                "questions": questions,
                "trajectory": [],
                "message_to_user": parsed.get("message_to_user", ""),
                "vlm_full": parsed,
            }
        else:
            plan = {
                "run_id": run_id,
                "goal": instruction,
                "status": "vlm_parse_failed",
                "user_raw": instruction,
                "intent_summary": "",
                "ambiguity": {},
                "image": {
                    "modality": {"label": "unknown", "confidence": None, "notes": ""},
                    "anatomy": {"region": "unknown", "confidence": None, "notes": ""},
                },
                "task_intent": {},
                "evidence": [],
                "questions": [],
                "trajectory": [],
                "message_to_user": "",
            }

        # ==================================================================
        # 3. LLM Client + tool registry
        # ==================================================================
        llm_base = os.environ.get("LLM_URL", "http://127.0.0.1:8002").rstrip("/")
        llm_client = LLMClient(llm_base) if llm_base else None
        tools_full = list_tools()
        tools_compact = tool_summary_for_llm(tools_full)

        # ==================================================================
        # 4. Planner — tool + target selection via LLM
        # ==================================================================
        planner_out = {}
        t_plan0 = time.time()
        if llm_client and parsed_found:
            try:
                log("Calling PlannerAgent")
                planner_out = PlannerAgent(llm_client).run(
                    plan.get("vlm_full", plan),
                    tools_compact=tools_compact,
                    user_instruction=instruction,
                )
                plan["trajectory"] = planner_out.get("steps", [])
                plan["planner_selected_tool"] = planner_out.get("selected_tool")
                trace["planner"] = planner_out
            except Exception as pe:
                log("PlannerAgent failed", error=str(pe))
                trace["planner"] = {"error": str(pe)}
        trace["timing"]["planner_s"] = float(time.time() - t_plan0)

        # ==================================================================
        # 5. Critic — validate / correct Planner decision via LLM
        # ==================================================================
        trace["critic"]["tool_registry"] = tools_compact
        t_crit0 = time.time()
        critic_raw: Dict[str, Any] = {}
        decision: Dict[str, Any] = {
            "tool": None, "action": "error",
            "reason": "CriticAgent could not be called",
            "questions": [], "params": {},
        }

        if llm_client:
            try:
                log("Calling CriticAgent")
                td, critic_raw = CriticAgent(llm_client).run(
                    planner_output=planner_out,
                    vlm_json=plan.get("vlm_full", plan) if parsed_found else {},
                    tools_compact=tools_compact,
                    user_instruction=instruction,
                )
                if td and td.tool:
                    decision = {
                        "tool": td.tool,
                        "action": "proceed",
                        "reason": td.reason,
                        "questions": critic_raw.get("questions", []),
                        "params": td.params if td.params else {},
                    }
                elif td and hasattr(td, 'action') and td.action == "need_clarification":
                    decision = {
                        "tool": td.tool,
                        "action": "need_clarification",
                        "reason": td.reason,
                        "questions": critic_raw.get("questions", getattr(td, 'questions', [])),
                        "params": td.params if td.params else {},
                    }
                else:
                    decision = {
                        "tool": None,
                        "action": "error",
                        "reason": f"CriticAgent returned no tool: {td.reason if td else 'no decision'}",
                        "questions": [],
                        "params": {},
                    }
                trace["critic"]["source"] = "llm"
            except Exception as ce:
                log("CriticAgent failed", error=str(ce))
                decision = {
                    "tool": None,
                    "action": "error",
                    "reason": f"CriticAgent exception: {ce}",
                    "questions": [],
                    "params": {},
                }
                trace["critic"]["error"] = str(ce)
        else:
            decision = {
                "tool": None, "action": "error",
                "reason": "LLM client unavailable",
                "questions": [], "params": {},
            }

        trace["timing"]["critic_s"] = float(time.time() - t_crit0)
        trace["critic"]["decision"] = decision
        plan["decision"] = decision
        plan["questions"] = decision.get("questions", plan.get("questions", []))

        # ==================================================================
        # 6. Explainer — generate clarification message if needed
        # ==================================================================
        explainer_text = ""
        t_exp0 = time.time()
        if llm_client:
            try:
                explainer_text = ExplainerAgent(llm_client).run(
                    planner_json=plan.get("vlm_full", plan) if parsed_found else {},
                    critic_raw=critic_raw,
                    user_override_requested=user_override
                )
                trace["explainer"]["ok"] = True
            except Exception as ee:
                trace["explainer"]["ok"] = False
                trace["explainer"]["error"] = str(ee)
        trace["timing"]["explainer_s"] = float(time.time() - t_exp0)
        if explainer_text:
            trace["explainer"]["text_head"] = explainer_text[:1200]

        # ==================================================================
        # 7. User override — re-ask Critic if user clicked "Proceed anyway"
        # ==================================================================
        if decision["action"] == "need_clarification" and user_override:
            log("User override — asking CriticAgent again")
            try:
                td_override, critic_raw_override = CriticAgent(llm_client).run(
                    planner_output=planner_out,
                    vlm_json=plan.get("vlm_full", plan) if parsed_found else {},
                    tools_compact=tools_compact,
                    user_instruction=instruction,
                )
                if td_override and td_override.tool:
                    decision["tool"] = td_override.tool
                    decision["action"] = "proceed"
                    decision["params"] = td_override.params if td_override.params else {}
                    decision["reason"] = (decision.get("reason", "") + " | USER OVERRIDE").strip()
                    decision["questions"] = critic_raw_override.get("questions", [])
                    trace["critic"]["override"] = True
                else:
                    decision["action"] = "error"
                    decision["reason"] = "CriticAgent could not select tool even with user override"
            except Exception as oe:
                log("CriticAgent override failed", error=str(oe))
                decision["action"] = "error"
                decision["reason"] = f"CriticAgent override exception: {oe}"

        # ==================================================================
        # 8. Executor — call the selected tool service
        # ==================================================================
        tool_result = None
        mask_dropdown_choices = []

        if decision["action"] == "proceed" and decision.get("tool"):
            t_exe0 = time.time()
            tool_name = str(decision["tool"]).strip().lower()
            tool_url = os.environ.get(
                f"TOOL_URL_{tool_name.upper()}",
                DEFAULT_TOOL_URLS.get(tool_name, ""),
            ).rstrip("/")
            trace["executor"]["tool"] = tool_name
            trace["executor"]["tool_url"] = tool_url

            if tool_url:
                try:
                    out_path = os.path.join(
                        os.environ.get("OUTPUT_DIR", "/tmp/outputs"),
                        run_id, tool_name,
                    )
                    os.makedirs(out_path, exist_ok=True)
                    params = dict(decision.get("params", {}))

                    # -- VoxTell: normalise prompt parameter -----------------
                    if tool_name == "voxtell":
                        if not params.get("prompts") and not params.get("text_prompt"):
                            params["prompts"] = str(
                                (plan.get("task_intent") or {}).get("target", instruction)
                            ).strip()
                        elif params.get("text_prompt") and not params.get("prompts"):
                            params["prompts"] = params.pop("text_prompt")

                    # -- BiomedParse: inject modality + text_prompt ----------
                    if tool_name == "biomedparse":
                        if "modality" not in params:
                            raw_mod = str(
                                plan.get("image", {}).get("modality", {}).get("label", "CT")
                            ).upper()
                            _MRI_ALIASES = {"T1", "T1-WEIGHTED", "T2", "T2-WEIGHTED", "FLAIR", "DWI", "MR", "MRI"}
                            _CT_ALIASES = {"CECT", "NCCT", "CT"}
                            if raw_mod in _MRI_ALIASES or "MRI" in raw_mod or "MR" in raw_mod:
                                params["modality"] = "MRI"
                            elif raw_mod in _CT_ALIASES or "CT" in raw_mod:
                                params["modality"] = "CT"
                            else:
                                params["modality"] = raw_mod
                        if not params.get("text_prompt"):
                            params["text_prompt"] = str(
                                (plan.get("task_intent") or {}).get("target", instruction)
                            ).strip()

                    # -- Copy input to scratch (Gradio tmp not accessible) ---
                    scratch_input = os.path.join(
                        os.environ.get("OUTPUT_DIR", "/tmp/outputs"),
                        run_id,
                        "input_" + os.path.basename(nifti_path),
                    )
                    os.makedirs(os.path.dirname(scratch_input), exist_ok=True)
                    shutil.copy2(nifti_path, scratch_input)

                    payload = {
                        "input_path": scratch_input,
                        "output_path": out_path,
                        "params": params,
                    }
                    log("Calling tool", url=f"{tool_url}/infer", tool=tool_name, params=params)
                    r = requests.post(f"{tool_url}/infer", json=payload, timeout=600)
                    r.raise_for_status()
                    out = r.json()
                    tool_result = {
                        "tool": tool_name,
                        "ok": out.get("ok", out.get("status") == "success"),
                        "message": out.get("message", "ok"),
                        "mask_path": out.get("mask_path", out.get("output_path")),
                        "structures": out.get("structures", []),
                        "extra": {},
                    }
                    trace["executor"]["ok"] = True

                    # -- Load mask files into session state for overlay viewer
                    mask_dir = tool_result.get("mask_path")
                    state["masks"] = {}
                    state["tool_name"] = tool_name

                    if mask_dir and os.path.isdir(mask_dir):
                        # TotalSeg / VoxTell: directory of .nii.gz files
                        nii_files = sorted([
                            f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")
                        ])
                        state["mask_dir"] = mask_dir
                        for f in nii_files:
                            name = f.replace(".nii.gz", "")
                            try:
                                m = nib.load(os.path.join(mask_dir, f)).get_fdata()
                                state["masks"][name] = m
                            except Exception:
                                pass

                    elif mask_dir and os.path.isfile(mask_dir) and mask_dir.endswith((".nii.gz", ".nii")):
                        # BiomedParse: single .nii.gz output
                        state["mask_dir"] = os.path.dirname(mask_dir)
                        name = os.path.basename(mask_dir).replace(".nii.gz", "").replace(".nii", "")
                        try:
                            m = nib.load(mask_dir).get_fdata()
                            state["masks"][name] = m
                        except Exception:
                            pass

                    elif mask_dir and os.path.isfile(mask_dir) and mask_dir.endswith(".npz"):
                        # BiomedParse fallback: .npz output
                        state["mask_dir"] = os.path.dirname(mask_dir)
                        name = os.path.basename(mask_dir).replace(".npz", "")
                        try:
                            npz = np.load(mask_dir, allow_pickle=True)
                            mask_key = next(
                                (k for k in ['mask', 'seg', 'segs', 'pred', 'arr_0'] if k in npz),
                                list(npz.keys())[0],
                            )
                            m = npz[mask_key]
                            if m.ndim == 4:
                                m = m[0]
                            state["masks"][name] = m
                        except Exception:
                            pass

                    mask_dropdown_choices = list(state["masks"].keys())

                except Exception as ee:
                    trace["executor"]["ok"] = False
                    trace["executor"]["error"] = str(ee)
                    tool_result = {
                        "tool": tool_name, "ok": False,
                        "message": f"Tool call failed: {ee}",
                        "mask_path": None, "extra": {},
                    }
            else:
                tool_result = {
                    "tool": tool_name, "ok": False,
                    "message": f"Tool URL not found for {tool_name}",
                    "mask_path": None, "extra": {},
                }
                trace["executor"]["ok"] = False

            trace["timing"]["executor_s"] = float(time.time() - t_exe0)
            trace["executor"]["tool_result"] = tool_result
            plan["tool_result"] = tool_result

        # ==================================================================
        # 9. Verifier — deterministic mask quality check
        # ==================================================================
        if tool_result:
            try:
                verifier_out = VerifierAgent.run(tool_result, plan)
                trace["verifier"] = verifier_out
                plan["verification"] = verifier_out
                if not verifier_out["ok"]:
                    tool_result["message"] += f" | VERIFICATION FAILED: {verifier_out['issues']}"
            except Exception as ve:
                trace["verifier"] = {"error": str(ve)}

        # ==================================================================
        # 10. Reporter — generate final summary text via LLM
        # ==================================================================
        t_rep0 = time.time()
        reporter_text = ""
        if tool_result and llm_client:
            try:
                reporter_text = ReporterAgent(llm_client).run(
                    planner_json=plan.get("vlm_full", plan) if parsed_found else {},
                    tool_result=tool_result,
                )
                trace["reporter"]["ok"] = True
            except Exception as re_:
                trace["reporter"]["ok"] = False
                trace["reporter"]["error"] = str(re_)
        trace["timing"]["reporter_s"] = float(time.time() - t_rep0)
        trace["timing"]["total_s"] = float(time.time() - t0)

        # ==================================================================
        # 11. Save plan and trace artefacts to OUTPUT_DIR
        # ==================================================================
        out_dir = os.environ.get(
            "OUTPUT_DIR",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs")),
        )
        os.makedirs(out_dir, exist_ok=True)
        save_json(os.path.join(out_dir, f"plan_{run_id}.json"), plan)
        save_json(os.path.join(out_dir, f"trace_{run_id}.json"), trace)

        # ==================================================================
        # 12. Build the report string shown in the UI
        # ==================================================================
        report = _build_report(
            plan, decision, trace, resp, vlm_text, parsed_found,
            explainer_text, reporter_text, tool_result, mask_dropdown_choices,
        )

        overlay_img = img2d_u8
        if mask_dropdown_choices:
            overlay_img = on_mask_select(state, axis, z, mask_dropdown_choices[0])

        return (
            state, img2d_u8, overlay_img, report,
            _safe_json_dumps(plan, "plan"),
            _safe_json_dumps(trace, "trace"),
            gr.update(
                choices=mask_dropdown_choices,
                value=mask_dropdown_choices[0] if mask_dropdown_choices else None,
            ),
        )

    # -- Top-level exception handler ----------------------------------------
    except Exception as e:
        trace["error"] = f"{type(e).__name__}: {e}"
        trace["traceback"] = traceback.format_exc()[:4000]
        trace["timing"]["total_s"] = float(time.time() - t0)
        error_report = (
            f"❌ PIPELINE ERROR: {type(e).__name__}: {e}\n\n"
            f"{traceback.format_exc()[:2000]}"
        )

        # Build a meaningful error plan (never empty)
        error_plan = {
            "run_id": run_id if 'run_id' in dir() else "unknown",
            "status": "pipeline_error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "instruction": str(instruction).strip() if instruction else "",
            "nifti_path": nifti_path if 'nifti_path' in dir() else "unknown",
            "stage_reached": (
                trace.get("events", [{}])[-1].get("msg", "unknown")
                if trace.get("events") else "init"
            ),
        }
        # Include any partial plan data collected before the crash
        if 'plan' in dir() and isinstance(plan, dict):
            error_plan["partial_plan"] = {
                k: v for k, v in plan.items()
                if k not in ("vlm_full", "vol") and not isinstance(v, np.ndarray)
            }

        return (
            state, img2d_u8, None,
            error_report,
            _safe_json_dumps(error_plan, "error_plan"),
            _safe_json_dumps(trace, "error_trace"),
            gr.update(choices=[], value=None),
        )


# ===========================================================================
# Gradio UI definition
# ===========================================================================

with gr.Blocks(title="Agentic Medical Image Segmentation",
               theme=gr.themes.Base(
                   primary_hue=gr.themes.colors.cyan,
                   secondary_hue=gr.themes.colors.orange,
                   neutral_hue=gr.themes.colors.gray,
                   font=gr.themes.GoogleFont("Source Sans Pro"),
               ).set(
                   body_background_fill="#0f1117",
                   body_background_fill_dark="#0f1117",
                   body_text_color="#e0e0e0",
                   body_text_color_dark="#e0e0e0",
                   block_background_fill="#1a1d27",
                   block_background_fill_dark="#1a1d27",
                   block_border_color="#2a2d37",
                   block_border_color_dark="#2a2d37",
                   block_label_text_color="#9ca3af",
                   block_label_text_color_dark="#9ca3af",
                   block_title_text_color="#e0e0e0",
                   block_title_text_color_dark="#e0e0e0",
                   input_background_fill="#232730",
                   input_background_fill_dark="#232730",
                   input_border_color="#3a3d47",
                   input_border_color_dark="#3a3d47",
                   button_primary_background_fill="#5C9EB8",
                   button_primary_background_fill_dark="#5C9EB8",
                   button_primary_text_color="#ffffff",
                   button_primary_text_color_dark="#ffffff",
                   button_secondary_background_fill="#2a2d37",
                   button_secondary_background_fill_dark="#2a2d37",
                   button_secondary_text_color="#e0e0e0",
                   button_secondary_text_color_dark="#e0e0e0",
               )) as demo:

    gr.Markdown(
        """
        <div style="text-align:center; padding: 10px 0 5px 0;">
            <h1 style="margin:0; color:#5C9EB8; font-size:28px; letter-spacing:1px;">
                Agentic Medical Image Segmentation
            </h1>
            <p style="color:#9ca3af; font-size:13px; margin-top:4px;">
                VLM &rarr; Planner &rarr; Critic &rarr; Explainer &rarr; Tool Executor &rarr; Verifier &rarr; Reporter
            </p>
            <div style="display:flex; justify-content:center; gap:20px; margin-top:8px; font-size:11px;">
                <span><span style="color:#5C9EB8;">&#9632;</span> TotalSeg</span>
                <span><span style="color:#E07A5F;">&#9632;</span> VoxTell</span>
                <span><span style="color:#F2CC8F;">&#9632;</span> BiomedParse</span>
                <span><span style="color:#81B29A;">&#9632;</span> Pipeline</span>
            </div>
        </div>
        """
    )

    state = gr.State({})

    with gr.Row():
        with gr.Column(scale=2):
            nifti = gr.File(label="Upload NIfTI (.nii/.nii.gz)", file_types=[".nii", ".gz"])
            load_btn = gr.Button("Load Volume", variant="primary")
            status = gr.Textbox(label="Status", value="", interactive=False)
            axis = gr.Radio(choices=[0, 1, 2], value=2, label="Axis (0=sagittal, 1=coronal, 2=axial)")
            z = gr.Slider(0, 10, value=0, step=1, label="Slice", interactive=True)
            mode = gr.Radio(choices=["3D volume", "Multi-slice", "2D slice"], value="3D volume", label="Input mode")
            n_slices = gr.Slider(8, 256, value=32, step=1, label="Num slices (3D/Multi)")
            instruction = gr.Textbox(label="Goal / Instruction", value="Segment lung infection", lines=2)
            run_btn = gr.Button("Run Pipeline", variant="primary")

        with gr.Column(scale=2):
            preview_img = gr.Image(label="CT Preview", type="numpy", height=380)
            mask_dropdown = gr.Dropdown(label="Select mask to overlay", choices=[], value=None, interactive=True)
            overlay = gr.Image(label="Overlay (CT + Mask)", type="numpy", height=380)

    with gr.Row(visible=False) as clarification_row:
        gr.Markdown("### ⚠️ Clarification needed — see Report below")
        proceed_btn = gr.Button("Proceed anyway", variant="secondary")
        cancel_btn = gr.Button("Cancel", variant="stop")

    with gr.Row():
        report = gr.Textbox(label="Agent Report", lines=22)
    with gr.Accordion("Plan JSON", open=False):
        plan_json = gr.Code(language="json")
    with gr.Accordion("Trace JSON", open=False):
        trace_json = gr.Code(language="json")

    # -- Event bindings -----------------------------------------------------

    load_btn.click(
        fn=on_load_nifti, inputs=[nifti],
        outputs=[state, preview_img, z, axis, status, mask_dropdown],
    )
    axis.change(
        fn=on_z_or_axis_change, inputs=[state, axis, z, mask_dropdown],
        outputs=[preview_img, z, overlay],
    )
    z.change(
        fn=on_z_or_axis_change, inputs=[state, axis, z, mask_dropdown],
        outputs=[preview_img, z, overlay],
    )
    mask_dropdown.change(
        fn=on_mask_select, inputs=[state, axis, z, mask_dropdown],
        outputs=[overlay],
    )

    # Toggle the clarification banner based on report content
    def _check_clarification(report_text):
        show = "action: need_clarification" in str(report_text)
        return gr.update(visible=show)

    def on_proceed(state, axis, z, mode, n_slices, instruction):
        return on_run(state, axis, z, mode, n_slices, instruction, user_override=True)

    def on_cancel(state, axis, z):
        preview, z_upd = on_preview_change(state, axis, z)
        _e = {"status": "cancelled", "reason": "User cancelled the pipeline run."}
        return (
            state, preview, None,
            "Cancelled.",
            _safe_json_dumps(_e, "plan"),
            _safe_json_dumps(_e, "trace"),
            gr.update(choices=[], value=None),
            gr.update(visible=False),
        )

    run_btn.click(
        fn=on_run,
        inputs=[state, axis, z, mode, n_slices, instruction],
        outputs=[state, preview_img, overlay, report, plan_json, trace_json, mask_dropdown],
    ).then(
        fn=_check_clarification,
        inputs=[report],
        outputs=[clarification_row],
    )

    proceed_btn.click(
        fn=on_proceed,
        inputs=[state, axis, z, mode, n_slices, instruction],
        outputs=[state, preview_img, overlay, report, plan_json, trace_json, mask_dropdown],
    ).then(
        fn=lambda *a: gr.update(visible=False),
        inputs=[], outputs=[clarification_row],
    )

    cancel_btn.click(
        fn=on_cancel, inputs=[state, axis, z],
        outputs=[state, preview_img, overlay, report, plan_json, trace_json, mask_dropdown, clarification_row],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
