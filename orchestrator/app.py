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
#                  selects the best tool + target.
#   3. Critic    — CriticAgent validates the Planner's decision; falls back to
#                  _deterministic_critic() if the LLM service is unavailable.
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
import os
import time
import json
import traceback
from typing import Optional, Tuple, Dict, Any, List
import gradio as gr
import numpy as np
import nibabel as nib

from contracts import (now_run_id, Trace, Evidence, Plan, ToolDecision, ToolResult, save_json)
from clients import VLMClient, LLMClient, ToolClient
from agents import CriticAgent, ExplainerAgent, ReporterAgent, PlannerAgent, VerifierAgent
from tool_registry import list_tools, tool_summary_for_llm, get_totalseg_structures


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
    """Extract a 2-D slice at index *z* along *axis* (0=sagittal,1=coronal,2=axial)."""
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
    """Read a file from disk and return its base64-encoded string."""
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


def _map_to_totalseg_structures(target: str) -> List[str]:
    """Map a free-text target string to matching TotalSegmentator structure names.

    Tokenises *target*, then returns every structure whose name contains at
    least one of the resulting keywords (length > 2).
    Returns an empty list when nothing matches (caller should run full segmentation).
    """
    all_structures = get_totalseg_structures()
    words = [
        w for w in
        target.lower().replace("(", "").replace(")", "").replace("-", "_").replace(" ", "_").split("_")
        if len(w) > 2
    ]
    matched = [s for s in all_structures if any(w in s for w in words)]
    return matched if matched else []


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
    base = os.environ.get("VLM_URL", "http://localhost:8000").rstrip("/")
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
    return r.json()


# ---------------------------------------------------------------------------
# Deterministic critic fallback
# ---------------------------------------------------------------------------

def _deterministic_critic(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based tool selection used when the LLM Critic is unavailable.

    Applies hard-coded rules (modality → tool, lesion keywords → voxtell,
    anatomy keywords in CT → totalseg, etc.) and returns the same dict shape
    as CriticAgent.run() would.
    """
    mod = (plan.get("image", {}).get("modality", {}) or {})
    task = (plan.get("task_intent", {}) or {})
    modality_label = str(mod.get("label", "unknown"))
    modality_conf = mod.get("confidence", None)
    task_type = str(task.get("type", "unknown"))
    task_target = str(task.get("target", "unknown"))
    questions_out = list(plan.get("questions", []) or [])
    need_clar = False
    reasons = []
    CONF_MIN = 0.55
    mod_norm = modality_label.strip().upper()
    if mod_norm in ("XR", "X-RAY", "XRAY", "X-RAY/CT"):
        mod_norm = "XRAY"
    elif mod_norm in ("US", "ULTRASOUND"):
        mod_norm = "US"
    elif mod_norm not in ("CT", "MRI", "PET", "XRAY", "US"):
        mod_norm = "Unknown"
    if str(plan.get("status", "")) == "need_clarification":
        need_clar = True
        reasons.append("planner_status_need_clarification")
    if mod_norm == "Unknown":
        need_clar = True
        reasons.append("modality_unknown")
        questions_out.append("What is the imaging modality (CT/MRI/PET/X-ray/Ultrasound)?")
    elif modality_conf is not None and float(modality_conf) < CONF_MIN:
        need_clar = True
        reasons.append("modality_low_confidence")
        questions_out.append("Please confirm the imaging modality.")
    if task_type.lower() in ("unknown", "", "none"):
        need_clar = True
        reasons.append("task_type_unknown")
        questions_out.append("What do you want to do: segmentation, localization, or diagnosis?")
    if task_target.lower() in ("unknown", "", "none", "something"):
        need_clar = True
        reasons.append("target_unknown")
        questions_out.append("What structure/pathology should be segmented?")
    decision: Dict[str, Any] = {
        "tool": None,
        "action": "need_clarification" if need_clar else "proceed",
        "reason": ";".join(reasons) if reasons else "ok",
        "questions": questions_out,
        "params": {},
    }
    if not need_clar:
        tgt = task_target.lower()
        lesion_kw = ("tumor", "lesion", "infection", "opacity", "nodule", "mass", "consolidation", "ggo", "ground-glass")
        anatomy_kw = ("lung", "liver", "kidney", "heart", "brain", "spleen", "pancreas", "aorta", "vertebra", "hippocampus", "myocardium", "ventricle")
        _mri_norm = mod_norm == "MRI" or any(x in modality_label.upper() for x in ("MRI", "T1", "T2", "FLAIR", "DWI"))
        if any(w in tgt for w in lesion_kw):
            # Lesion / pathology targets always go to VoxTell
            decision["tool"] = "voxtell"
            decision["reason"] = "lesion_like_target_voxtell"
            decision["params"] = {"prompts": task_target}
        elif _mri_norm:
            # MRI 3-D volumes go to VoxTell (BiomedParse is 2-D only)
            decision["tool"] = "voxtell"
            decision["reason"] = "mri_modality_voxtell"
            decision["params"] = {"prompts": task_target}
        elif any(w in tgt for w in anatomy_kw) and mod_norm == "CT":
            decision["tool"] = "totalseg"
            decision["reason"] = "anatomy_like_target_ct"
            decision["params"] = {"structure": task_target, "mode": "3d"}
        else:
            decision["tool"] = "voxtell"
            decision["reason"] = "fallback_voxtell"
            decision["params"] = {"prompts": task_target}
    return decision


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def _make_overlay(ct_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
    """Composite a red mask onto a greyscale CT slice and return an RGB array."""
    ct_rgb = np.stack([_normalize_to_uint8(ct_slice)] * 3, axis=-1)
    mask_bin = (mask_slice > 0).astype(np.uint8)
    overlay = ct_rgb.copy()
    overlay[mask_bin == 1, 0] = 255
    overlay[mask_bin == 1, 1] = 0
    overlay[mask_bin == 1, 2] = 0
    return overlay


# ---------------------------------------------------------------------------
# Gradio event handlers
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
    st = {"path": path, "shape": list(vol.shape), "vol": vol, "mask_dir": None, "masks": {}}
    z_slider = gr.update(minimum=0, maximum=zmax, value=z0, step=1, interactive=True)
    return st, preview, z_slider, gr.update(value=axis), f"Loaded: {os.path.basename(path)} | shape={vol.shape}", gr.update(choices=[], value=None)


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
    zmax = _max_index(vol, axis)
    z = int(np.clip(z, 0, zmax))
    ct_slice = _slice_from_vol(vol, axis, z)
    if selected_mask in masks:
        mask_vol = masks[selected_mask]
        if mask_vol.shape == vol.shape:
            mask_slice = _slice_from_vol(mask_vol, axis, z)
            return _make_overlay(ct_slice, mask_slice)
    return _normalize_to_uint8(ct_slice)


def on_run(state, axis, z, mode, n_slices, instruction, user_override=False):
    """Main pipeline handler: VLM → Planner → Critic → Executor → Verifier → Reporter.

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
    if not state or "vol" not in state:
        return state, None, "Error: please upload and load a NIfTI file first.", "{}", "{}", gr.update(choices=[], value=None)
    if not instruction or not str(instruction).strip():
        preview, _ = on_preview_change(state, axis, z)
        return state, preview, "Error: instruction is empty.", "{}", "{}", gr.update(choices=[], value=None)

    vol = state["vol"]
    run_id = now_run_id()
    t0 = time.time()
    zmax = _max_index(vol, axis)
    z = int(np.clip(z, 0, zmax))
    img2d_u8 = _normalize_to_uint8(_slice_from_vol(vol, axis, z))

    if mode != "3D volume":
        return state, img2d_u8, "Only '3D volume' mode is enabled.", "{}", "{}", gr.update(choices=[], value=None)

    nifti_path = state.get("path", "")
    if not nifti_path or not os.path.exists(nifti_path):
        return state, img2d_u8, "NIfTI path not found.", "{}", "{}", gr.update(choices=[], value=None)

    trace = {
        "run_id": run_id, "events": [], "timing": {},
        "inputs": {"path": nifti_path, "axis": int(axis), "z": int(z), "mode": mode, "n_slices": int(n_slices), "instruction": str(instruction).strip()},
        "vlm": {}, "planner": {}, "critic": {}, "explainer": {}, "executor": {}, "verifier": {}, "reporter": {},
    }

    def log(msg, **extra):
        e = {"t": time.time(), "msg": msg}
        if extra:
            e["extra"] = extra
        trace["events"].append(e)

    log("Run clicked")

    try:
        # ------------------------------------------------------------------
        # VLM — image understanding
        # ------------------------------------------------------------------
        t_vlm0 = time.time()
        resp = call_hulu_infer(nifti_path, axis, n_slices, instruction)
        trace["timing"]["vlm_s"] = float(time.time() - t_vlm0)
        vlm_text = str(resp.get("text", ""))
        trace["vlm"]["response_meta"] = {"model_id": resp.get("model_id"), "device": resp.get("device")}
        trace["vlm"]["text_head"] = vlm_text[:2000]
        parsed = _extract_first_json_obj(vlm_text)
        parsed_found = isinstance(parsed, dict)
        trace["vlm"]["parsed_json_found"] = parsed_found

        # ------------------------------------------------------------------
        # Plan build — normalise VLM output into a canonical plan dict
        # ------------------------------------------------------------------
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
                "run_id": run_id, "goal": instruction, "status": parsed.get("status", "ok"),
                "user_raw": user_blk.get("raw_input", instruction), "intent_summary": user_blk.get("intent_summary", ""),
                "ambiguity": user_blk.get("ambiguity", {}) if isinstance(user_blk.get("ambiguity", {}), dict) else {},
                "image": {
                    "modality": {"label": mod_blk.get("label", "unknown"), "confidence": mod_blk.get("confidence", None), "notes": mod_blk.get("notes", "")},
                    "anatomy": {"region": anat_blk.get("region", "unknown"), "confidence": anat_blk.get("confidence", None), "notes": anat_blk.get("notes", "")},
                },
                "task_intent": task, "evidence": evidence, "questions": questions,
                "trajectory": [], "message_to_user": parsed.get("message_to_user", ""), "vlm_full": parsed,
            }
        else:
            plan = {
                "run_id": run_id, "goal": instruction, "status": "need_clarification",
                "user_raw": instruction, "intent_summary": "", "ambiguity": {},
                "image": {"modality": {"label": "unknown", "confidence": None, "notes": ""}, "anatomy": {"region": "unknown", "confidence": None, "notes": ""}},
                "task_intent": {}, "evidence": [], "questions": ["Planner output was not valid JSON."],
                "trajectory": [], "message_to_user": "",
            }

        llm_base = os.environ.get("LLM_URL", "").rstrip("/")
        llm_client = LLMClient(llm_base) if llm_base else None

        # Tool registry for Planner and Critic
        tools_full = list_tools()
        tools_compact = tool_summary_for_llm(tools_full)

        # ------------------------------------------------------------------
        # Planner — tool + target selection via LLM
        # ------------------------------------------------------------------
        t_plan0 = time.time()
        planner_out: Dict[str, Any] = {}
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

        # ------------------------------------------------------------------
        # Critic — validate / correct Planner decision
        # ------------------------------------------------------------------
        trace["critic"]["tool_registry"] = tools_compact
        t_crit0 = time.time()
        critic_raw: Dict[str, Any] = {}
        if llm_client:
            try:
                log("Calling CriticAgent")
                td, critic_raw = CriticAgent(llm_client).run(
                    planner_output=planner_out,
                    vlm_json=plan.get("vlm_full", plan),
                    tools_compact=tools_compact,
                    user_instruction=instruction,
                )
                decision = {
                    "tool": td.tool,
                    "action": "need_clarification" if td.tool is None else "proceed",
                    "reason": td.reason,
                    "questions": critic_raw.get("questions", []),
                    "params": td.params,
                }
                trace["critic"]["source"] = "llm"
            except Exception as ce:
                log("CriticAgent failed", error=str(ce))
                trace["critic"]["llm_error"] = f"{type(ce).__name__}: {ce}"
                decision = _deterministic_critic(plan)
                trace["critic"]["source"] = "deterministic_fallback"
        else:
            decision = _deterministic_critic(plan)
            trace["critic"]["source"] = "deterministic_no_llm"
        trace["timing"]["critic_s"] = float(time.time() - t_crit0)
        trace["critic"]["decision"] = decision
        plan["decision"] = decision
        plan["questions"] = decision.get("questions", plan.get("questions", []))

        # ------------------------------------------------------------------
        # Explainer — generate clarification message if needed
        # ------------------------------------------------------------------
        explainer_text = ""
        if decision["action"] == "need_clarification":
            t_exp0 = time.time()
            if llm_client:
                try:
                    explainer_text = ExplainerAgent(llm_client).run(
                        planner_output=plan.get("vlm_full", plan),
                        critic_raw=critic_raw,
                        user_override_requested=user_override,
                    )
                    trace["explainer"]["ok"] = True
                    trace["explainer"]["source"] = "llm"
                except Exception as ee:
                    trace["explainer"]["ok"] = False
                    trace["explainer"]["error"] = str(ee)
            trace["timing"]["explainer_s"] = float(time.time() - t_exp0)
            if not explainer_text:
                qs = plan.get("questions", []) or []
                bullets = "\n".join([f"  ? {q}" for q in qs]) if qs else "  ? Please clarify your request."
                explainer_text = f"Need clarification.\nReasons: {decision.get('reason', '')}\nQuestions:\n{bullets}"
                trace["explainer"]["source"] = "deterministic_fallback"
            trace["explainer"]["text_head"] = explainer_text[:1200]

        # ------------------------------------------------------------------
        # Executor — call the selected tool service
        # ------------------------------------------------------------------
        # If user clicked "Proceed anyway", force execution even when the
        # Critic said need_clarification and no tool was selected.
        if user_override and decision["action"] == "need_clarification" and decision.get("tool") is None:
            fallback = _deterministic_critic(plan)
            if fallback.get("tool"):
                decision["action"] = "proceed"
                decision["tool"] = fallback["tool"]
                decision["params"] = fallback.get("params", {})
                decision["reason"] = decision["reason"] + " [USER OVERRIDE — proceeding despite warning]"

        tool_result = None
        mask_dropdown_choices = []
        if decision["action"] == "proceed" and decision.get("tool"):
            t_exe0 = time.time()
            tool_name = str(decision["tool"]).strip().lower()
            tool_url = os.environ.get(f"TOOL_URL_{tool_name.upper()}", "").rstrip("/")
            trace["executor"]["tool"] = tool_name
            trace["executor"]["tool_url"] = tool_url

            if tool_url:
                try:
                    out_path = os.path.join(os.environ.get("OUTPUT_DIR", "/tmp/outputs"), run_id, tool_name)
                    os.makedirs(out_path, exist_ok=True)
                    params = dict(decision.get("params", {}))

                    # Map free-text target to exact TotalSegmentator structure names
                    if tool_name == "totalseg":
                        tgt = params.get("structure", (plan.get("task_intent") or {}).get("target", ""))
                        structures = _map_to_totalseg_structures(str(tgt))
                        if structures:
                            params["structures"] = structures
                            params.pop("structure", None)
                            log(f"TotalSeg structures: {structures}")
                        else:
                            params.pop("structure", None)
                            log("TotalSeg: no match, running full segmentation")

                    # Normalise VoxTell prompt parameter
                    if tool_name == "voxtell":
                        if not params.get("prompts") and not params.get("text_prompt"):
                            params["prompts"] = str((plan.get("task_intent") or {}).get("target", instruction)).strip()
                        elif params.get("text_prompt") and not params.get("prompts"):
                            params["prompts"] = params.pop("text_prompt")

                    # Inject modality and text_prompt for BiomedParse
                    if tool_name == "biomedparse":
                        if "modality" not in params:
                            raw_mod = str(plan.get("image", {}).get("modality", {}).get("label", "CT")).upper()
                            _MRI_ALIASES = {"T1", "T1-WEIGHTED", "T2", "T2-WEIGHTED", "FLAIR", "DWI", "MR", "MRI"}
                            _CT_ALIASES = {"CECT", "NCCT", "CT"}
                            if raw_mod in _MRI_ALIASES or "MRI" in raw_mod or "MR" in raw_mod:
                                params["modality"] = "MRI"
                            elif raw_mod in _CT_ALIASES or "CT" in raw_mod:
                                params["modality"] = "CT"
                            else:
                                params["modality"] = raw_mod
                        if not params.get("text_prompt"):
                            params["text_prompt"] = str((plan.get("task_intent") or {}).get("target", instruction)).strip()

                    # BiomedParse runs inside a SIF container and cannot access
                    # Gradio's /gradio_tmp UUID paths — copy the input to scratch.
                    tool_input_path = nifti_path
                    if tool_name == "biomedparse" and "/gradio_tmp" in nifti_path:
                        import shutil
                        scratch_input = os.path.join(os.environ.get("OUTPUT_DIR", "/tmp/outputs"), run_id, "input_" + os.path.basename(nifti_path))
                        shutil.copy2(nifti_path, scratch_input)
                        tool_input_path = scratch_input
                        log(f"Biomedparse: copied input to {scratch_input}")

                    payload = {"input_path": tool_input_path, "output_path": out_path, "params": params}
                    log("Calling tool", url=f"{tool_url}/infer", tool=tool_name)
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

                    # Load mask files into session state for the overlay viewer
                    mask_dir = tool_result.get("mask_path")
                    state["masks"] = {}
                    if mask_dir and os.path.isdir(mask_dir):
                        # TotalSeg / VoxTell: directory of .nii.gz files
                        nii_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")])
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
                        # BiomedParse fallback: .npz output (when NIfTI conversion failed)
                        state["mask_dir"] = os.path.dirname(mask_dir)
                        name = os.path.basename(mask_dir).replace(".npz", "")
                        try:
                            import numpy as np_
                            npz = np_.load(mask_dir, allow_pickle=True)
                            mask_key = next((k for k in ['mask', 'seg', 'segs', 'pred', 'arr_0'] if k in npz), list(npz.keys())[0])
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
                    tool_result = {"tool": tool_name, "ok": False, "message": f"Tool call failed: {ee}", "mask_path": None, "extra": {}}
            else:
                tool_result = {"tool": tool_name, "ok": False, "message": f"Tool service URL not set. Define TOOL_URL_{tool_name.upper()} to enable execution.", "mask_path": None, "extra": {}}
                trace["executor"]["ok"] = False
                trace["executor"]["error"] = "tool_url_missing"

            trace["timing"]["executor_s"] = float(time.time() - t_exe0)
            trace["executor"]["tool_result"] = tool_result
            plan["tool_result"] = tool_result

        # ------------------------------------------------------------------
        # Verifier — deterministic mask quality check
        # ------------------------------------------------------------------
        if tool_result:
            try:
                verifier_out = VerifierAgent.run(tool_result, plan)
                trace["verifier"] = verifier_out
                plan["verification"] = verifier_out
                if not verifier_out["ok"]:
                    tool_result["message"] += f" | VERIFICATION FAILED: {verifier_out['issues']}"
            except Exception as ve:
                trace["verifier"] = {"error": str(ve)}

        # ------------------------------------------------------------------
        # Reporter — generate final summary text
        # ------------------------------------------------------------------
        t_rep0 = time.time()
        reporter_text = ""
        if tool_result and llm_client:
            try:
                reporter_text = ReporterAgent(llm_client).run(planner_json=plan.get("vlm_full", plan), tool_result=tool_result)
                trace["reporter"]["ok"] = True
                trace["reporter"]["source"] = "llm"
            except Exception as re_:
                trace["reporter"]["ok"] = False
                trace["reporter"]["error"] = str(re_)
        else:
            trace["reporter"]["ok"] = True
            trace["reporter"]["source"] = "skipped"
        trace["timing"]["reporter_s"] = float(time.time() - t_rep0)
        trace["timing"]["total_s"] = float(time.time() - t0)

        # Save plan and trace artefacts to OUTPUT_DIR
        out_dir = os.environ.get("OUTPUT_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs")))
        os.makedirs(out_dir, exist_ok=True)
        save_json(os.path.join(out_dir, f"plan_{run_id}.json"), plan)
        save_json(os.path.join(out_dir, f"trace_{run_id}.json"), trace)

        # ------------------------------------------------------------------
        # Build the report string shown in the UI
        # ------------------------------------------------------------------
        if not parsed_found:
            report = f"✅ VLM OK | model={resp.get('model_id')} | device={resp.get('device')}\nparsed_json_found=False\n\n{vlm_text[:1500]}"
        else:
            ev_lines = "\n".join([f"  - {e}" for e in (plan.get("evidence", []) or [])]) or "  (none)"
            q_lines = "\n".join([f"  ? {q}" for q in (plan.get("questions", []) or [])]) or "  (none)"
            tool_line = decision.get("tool") or "(none)"
            action_line = decision.get("action", "unknown")
            _mod = plan.get("image", {}).get("modality", {}) or {}
            _anat = plan.get("image", {}).get("anatomy", {}) or {}
            _task = plan.get("task_intent", {}) or {}
            traj_lines = ""
            if plan.get("trajectory"):
                traj_lines = "\nTRAJECTORY:\n" + "\n".join([f"  {s.get('step')}. [{s.get('action')}] {s.get('detail')}" for s in plan.get("trajectory", [])]) + "\n"
            toolres = plan.get("tool_result")
            toolres_line = ""
            if toolres:
                verif = plan.get("verification", {})
                verif_str = "OK" if verif.get("ok") else f"FAILED: {verif.get('issues', [])}"
                structs = toolres.get("structures", [])
                toolres_line = (
                    f"\n\nEXECUTOR:\n"
                    f"- tool: {toolres.get('tool')}\n"
                    f"- ok: {toolres.get('ok')}\n"
                    f"- structures: {structs}\n"
                    f"- message: {toolres.get('message')}\n"
                    f"- mask_path: {toolres.get('mask_path')}\n"
                    f"- verification: {verif_str}\n"
                )
                if mask_dropdown_choices:
                    toolres_line += f"- available masks: {mask_dropdown_choices}\n"
            explain_line = f"\n\nEXPLAINER:\n{explainer_text}" if action_line == "need_clarification" else ""
            reporter_line = f"\n\nREPORTER:\n{reporter_text}" if reporter_text else ""
            report = (
                f"✅ VLM OK | model={resp.get('model_id')} | device={resp.get('device')}\n"
                f"status={plan.get('status')}\n\n"
                f"USER:\n- raw: {plan.get('user_raw', '')}\n- intent_summary: {plan.get('intent_summary', '')}\n\n"
                f"IMAGE:\n- modality: {str(_mod.get('label','unknown')).upper()} (conf={_mod.get('confidence','?')})\n- anatomy: {_anat.get('region','unknown')} (conf={_anat.get('confidence','?')})\n\n"
                f"TASK:\n- type: {_task.get('type','unknown')}\n- target: {_task.get('target','unknown')}\n\n"
                f"EVIDENCE:\n{ev_lines}\n"
                f"{traj_lines}\n"
                f"CRITIC [{trace.get('critic',{}).get('source','unknown')}]:\n- action: {action_line}\n- tool: {tool_line}\n- reason: {decision.get('reason','')}\n\n"
                f"QUESTIONS:\n{q_lines}"
                f"{toolres_line}{explain_line}{reporter_line}"
            )

        return state, img2d_u8, report, json.dumps(plan, indent=2), json.dumps(trace, indent=2), gr.update(choices=mask_dropdown_choices, value=mask_dropdown_choices[0] if mask_dropdown_choices else None)

    except Exception as e:
        trace["vlm"]["error"] = f"{type(e).__name__}: {e}"
        trace["vlm"]["traceback"] = traceback.format_exc()[:4000]
        trace["timing"]["total_s"] = float(time.time() - t0)
        return state, img2d_u8, f"❌ VLM failed: {type(e).__name__}: {e}", "{}", json.dumps(trace, indent=2), gr.update(choices=[], value=None)


# ---------------------------------------------------------------------------
# Combined preview + overlay refresh helper
# ---------------------------------------------------------------------------

def on_z_or_axis_change(state, axis, z, mask_dropdown):
    """Refresh both the preview slice and the mask overlay when axis/z changes."""
    preview, z_update = on_preview_change(state, axis, z)
    overlay_img = on_mask_select(state, axis, z, mask_dropdown)
    return preview, z_update, overlay_img


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

with gr.Blocks(title="Agentic Segmentation", theme="Taithrah/Minimal") as demo:
    gr.Markdown("## Agentic Segmentation")
    state = gr.State({})
    with gr.Row():
        with gr.Column(scale=2):
            nifti = gr.File(label="Upload NIfTI (.nii/.nii.gz)", file_types=[".nii", ".gz"])
            load_btn = gr.Button("Load", variant="primary")
            status = gr.Textbox(label="Status", value="", interactive=False)
            axis = gr.Radio(choices=[0, 1, 2], value=2, label="Axis (0=sagittal, 1=coronal, 2=axial)")
            z = gr.Slider(0, 10, value=0, step=1, label="Slice z", interactive=True)
            mode = gr.Radio(choices=["3D volume", "Multi-slice", "2D slice"], value="3D volume", label="What to send to VLM?")
            n_slices = gr.Slider(8, 256, value=32, step=1, label="(3D/Multi-slice) num slices")
            instruction = gr.Textbox(label="Goal / Instruction", value="Segment lung infection", lines=2)
            run_btn = gr.Button("Run (VLM -> Planner -> Critic -> Tool)", variant="primary")
    with gr.Row(visible=False) as clarification_row:
        gr.Markdown("### ⚠️ Clarification needed — see Report below")
        proceed_btn = gr.Button("✅ Proceed anyway", variant="secondary")
        cancel_btn = gr.Button("❌ Cancel", variant="stop")
        with gr.Column(scale=2):
            preview_img = gr.Image(label="CT Preview", type="numpy", height=380)
            mask_dropdown = gr.Dropdown(label="Select mask to overlay", choices=[], value=None, interactive=True)
            overlay = gr.Image(label="Overlay (CT + Mask)", type="numpy", height=380)
    with gr.Row():
        report = gr.Textbox(label="Report", lines=18)
    with gr.Accordion("Plan JSON", open=False):
        plan_json = gr.Code(language="json")
    with gr.Accordion("Trace JSON", open=False):
        trace_json = gr.Code(language="json")

    load_btn.click(fn=on_load_nifti, inputs=[nifti], outputs=[state, preview_img, z, axis, status, mask_dropdown])
    axis.change(fn=on_z_or_axis_change, inputs=[state, axis, z, mask_dropdown], outputs=[preview_img, z, overlay])
    z.change(fn=on_z_or_axis_change, inputs=[state, axis, z, mask_dropdown], outputs=[preview_img, z, overlay])

    def _show_clarification_row(state, overlay, report, plan_json, trace_json, mask_dropdown):
        """Show the clarification row if the report signals need_clarification."""
        show = "Need clarification" in str(report) or "QUESTIONS" in str(report)
        return state, overlay, report, plan_json, trace_json, mask_dropdown, gr.update(visible=show)

    def on_proceed(state, axis, z, mode, n_slices, instruction):
        return on_run(state, axis, z, mode, n_slices, instruction, user_override=True)

    def on_cancel(state, axis, z):
        preview, z_upd = on_preview_change(state, axis, z)
        return state, preview, "Cancelled. Please adjust your instruction and try again.", "{}", "{}", gr.update(choices=[], value=None), gr.update(visible=False)

    run_btn.click(
        fn=on_run,
        inputs=[state, axis, z, mode, n_slices, instruction],
        outputs=[state, overlay, report, plan_json, trace_json, mask_dropdown],
    ).then(
        fn=_show_clarification_row,
        inputs=[state, overlay, report, plan_json, trace_json, mask_dropdown],
        outputs=[state, overlay, report, plan_json, trace_json, mask_dropdown, clarification_row],
    )

    proceed_btn.click(
        fn=on_proceed,
        inputs=[state, axis, z, mode, n_slices, instruction],
        outputs=[state, overlay, report, plan_json, trace_json, mask_dropdown],
    ).then(
        fn=lambda *a: gr.update(visible=False),
        inputs=[],
        outputs=[clarification_row],
    )

    cancel_btn.click(
        fn=on_cancel,
        inputs=[state, axis, z],
        outputs=[state, overlay, report, plan_json, trace_json, mask_dropdown, clarification_row],
    )
    mask_dropdown.change(fn=on_mask_select, inputs=[state, axis, z, mask_dropdown], outputs=[overlay])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))