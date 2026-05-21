# =============================================================================
# services/tool_biomedparse/api.py
#
# FastAPI wrapper around the BiomedParse v2 segmentation model.
# Supports CT, MRI, PET, Ultrasound and Microscopy inputs via text prompts.
#
# Two input modes
# ---------------
#   POST /segment  — multipart file upload (used for direct HTTP clients)
#   POST /infer    — path-based JSON request (used by the orchestrator)
#
# CT preprocessing
# ----------------
#   CT volumes are normalised using clinical window/level presets before
#   being passed to the model.  Available windows: soft_tissue, lung,
#   brain, bone.  Non-CT modalities use percentile clipping.
#
# Endpoints
# ---------
#   GET  /health          — liveness check
#   POST /segment         — file upload segmentation
#   POST /infer           — path-based segmentation (InferRequest → InferResponse)
#   GET  /mcp/tools/list  — MCP tool manifest
#   POST /mcp/tools/call  — MCP tool invocation wrapper
#
# Params (inside InferRequest.params)
# ------------------------------------
#   prompts   — list or comma-separated string of target structures
#   modality  — CT | MRI | PET | Ultrasound | Microscopy (default: CT)
#   window    — CT windowing preset (default: soft_tissue)
#
# Environment variables
# ---------------------
#   PYTORCH_CUDA_ALLOC_CONF  — set to expandable_segments:True at module load
# =============================================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.path.insert(0, "/app/BiomedParse")

import shutil
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from utils import process_input, process_output
from inference import postprocess, merge_multiclass_masks
from huggingface_hub import hf_hub_download

model = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CT windowing presets (window width W, window level L)
CT_WINDOWS = {
    "soft_tissue": {"W": 400,  "L": 40},
    "lung":        {"W": 1500, "L": -160},
    "brain":       {"W": 80,   "L": 40},
    "bone":        {"W": 1800, "L": 400},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load BiomedParse model at startup; release GPU memory on shutdown."""
    global model
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("Loading BiomedParse model...", flush=True)
    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir="/app/BiomedParse/configs/model", job_name="api")
    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")
    model.load_pretrained("/app/BiomedParse/weights/biomedparse_v2.ckpt")
    model = model.to(DEVICE).eval()
    print("Model loaded successfully.", flush=True)

    yield

    model = None
    torch.cuda.empty_cache()


app = FastAPI(title="BiomedParse Segmentation API", version="2.0", lifespan=lifespan)


# ----------------------------
# Preprocessing helpers
# ----------------------------

def preprocess_ct(volume: np.ndarray, window: str = "soft_tissue") -> np.ndarray:
    """Normalise CT HU values using a window/level preset and rescale to [0, 255]."""
    w = CT_WINDOWS.get(window, CT_WINDOWS["soft_tissue"])
    W, L = w["W"], w["L"]
    low = L - W / 2
    high = L + W / 2
    volume = np.clip(volume, low, high)
    volume = (volume - low) / (high - low) * 255.0
    return volume.astype(np.float32)


def preprocess_other(volume: np.ndarray) -> np.ndarray:
    """Clip to 0.5th–99.5th percentile and rescale to [0, 255] for non-CT modalities."""
    low = np.percentile(volume, 0.5)
    high = np.percentile(volume, 99.5)
    volume = np.clip(volume, low, high)
    if high > low:
        volume = (volume - low) / (high - low) * 255.0
    return volume.astype(np.float32)


def nifti_to_npz(
    nifti_path: str,
    modality: str = "CT",
    window: str = "soft_tissue",
    prompts: List[str] = None,
    tmp_dir: str = None,
) -> str:
    """Convert a NIfTI volume to the npz format expected by BiomedParse.

    Applies CT windowing or percentile clipping depending on modality,
    encodes text prompts as a numbered dict, and saves to tmp_dir/input.npz.
    """
    img = nib.load(nifti_path)
    volume = img.get_fdata().astype(np.float32)

    if modality.upper() == "CT":
        volume = preprocess_ct(volume, window)
    else:
        volume = preprocess_other(volume)

    text_prompts = {str(i+1): p for i, p in enumerate(prompts)}

    npz_path = os.path.join(tmp_dir, "input.npz")
    np.savez(npz_path, imgs=volume, text_prompts=text_prompts)
    return npz_path


def run_inference(npz_path: str) -> np.ndarray:
    """Run BiomedParse on a preprocessed npz file and return the mask array."""
    npz_data = np.load(npz_path, allow_pickle=True)
    imgs = npz_data["imgs"]
    text_prompts = npz_data["text_prompts"].item()

    ids = [int(k) for k in text_prompts.keys() if k != "instance_label"]
    ids.sort()
    text = "[SEP]".join([text_prompts[str(i)] for i in ids])

    imgs, pad_width, padded_size, valid_axis = process_input(imgs, 512)
    imgs = imgs.to(DEVICE).int()

    input_tensor = {
        "image": imgs.unsqueeze(0),
        "text": [text],
    }

    with torch.no_grad():
        output = model(input_tensor, mode="eval", slice_batch_size=4)

    mask_preds = output["predictions"]["pred_gmasks"]
    mask_preds = F.interpolate(
        mask_preds, size=(512, 512),
        mode="bicubic", align_corners=False, antialias=True,
    )
    mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
    mask_preds = merge_multiclass_masks(mask_preds, ids)
    mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)

    # process_output may return either a Tensor or a numpy array
    if hasattr(mask_preds, "cpu"):
        return mask_preds.cpu().numpy()
    return np.array(mask_preds)


def mask_to_nifti(mask: np.ndarray, reference_nifti_path: str, out_path: str):
    """Save a mask array as a NIfTI file, reusing the reference image's affine/header."""
    ref = nib.load(reference_nifti_path)
    nifti_img = nib.Nifti1Image(mask.astype(np.uint8), ref.affine, ref.header)
    nifti_img.header.set_data_dtype(np.uint8)
    nifti_img.header.set_qform(ref.affine, code=1)
    nifti_img.header.set_sform(ref.affine, code=1)
    nib.save(nifti_img, out_path)


def cleanup_temp_dir(path: str):
    shutil.rmtree(path, ignore_errors=True)


# ----------------------------
# Request / Response models
# ----------------------------

class InferRequest(BaseModel):
    input_path: str
    output_path: str
    params: dict = {}


class InferResponse(BaseModel):
    status: str
    output_path: Optional[str] = None
    prompts: Optional[List[str]] = None
    message: Optional[str] = None


class MCPCallRequest(BaseModel):
    name: str
    input: Dict[str, Any]


# ----------------------------
# Health
# ----------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "status": "ok",
        "tool": "BiomedParse",
        "version": "2.0",
    }


# ----------------------------
# File upload endpoint
# ----------------------------

@app.post("/segment")
async def segment_image(
    background_tasks: BackgroundTasks,
    prompts: List[str] = Form(...),
    modality: str = Form(default="CT"),
    window: str = Form(default="soft_tissue"),
    file: UploadFile = File(...),
):
    """Multipart upload endpoint — returns the segmentation mask as a .nii.gz file."""
    if not file.filename.endswith(".nii.gz") and not file.filename.endswith(".nii"):
        raise HTTPException(status_code=400, detail="File must be a .nii or .nii.gz NIfTI file")

    tmpdir = tempfile.mkdtemp()
    background_tasks.add_task(cleanup_temp_dir, tmpdir)

    raw_path = os.path.join(tmpdir, file.filename)
    out_path = os.path.join(tmpdir, "output.nii.gz")

    try:
        with open(raw_path, "wb") as buffer:
            buffer.write(await file.read())

        print(f"Prompts: {prompts}, modality: {modality}, window: {window}", flush=True)

        npz_path = nifti_to_npz(raw_path, modality, window, prompts, tmpdir)
        mask = run_inference(npz_path)

        print(f"Mask shape: {mask.shape}, unique: {np.unique(mask)}", flush=True)

        mask_to_nifti(mask, raw_path, out_path)

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

    return FileResponse(out_path, media_type="application/gzip", filename=f"seg_{file.filename}")


# ----------------------------
# Path-based endpoint (used by orchestrator)
# ----------------------------

@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest):
    """Path-based segmentation endpoint used by the orchestrator via ToolClient."""
    prompts = request.params.get("prompts", [])
    modality = request.params.get("modality", "CT")
    window = request.params.get("window", "soft_tissue")

    if isinstance(prompts, str):
        prompts = [prompts]
    if not prompts:
        raise HTTPException(status_code=400, detail="No prompts provided in params.")
    if not request.input_path.endswith((".nii.gz", ".nii")):
        raise HTTPException(status_code=400, detail="Invalid format. Only .nii or .nii.gz accepted.")

    os.makedirs(request.output_path, exist_ok=True)
    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(
        request.output_path,
        os.path.basename(request.input_path).replace(".nii.gz", "_seg.nii.gz"),
    )

    try:
        npz_path = nifti_to_npz(request.input_path, modality, window, prompts, tmpdir)
        mask = run_inference(npz_path)
        mask_to_nifti(mask, request.input_path, out_path)
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return InferResponse(status="success", output_path=out_path, prompts=prompts)


# ----------------------------
# MCP endpoints
# ----------------------------

@app.get("/mcp/tools/list")
def mcp_tools_list():
    return {
        "tools": [{
            "name": "biomedparse",
            "description": "3D biomedical image segmentation using BiomedParse v2. Supports CT, MRI, PET, Ultrasound and Microscopy with 200+ anatomical structures.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to input .nii or .nii.gz volume"},
                    "output_path": {"type": "string", "description": "Path to output directory"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "prompts": {"type": "array", "items": {"type": "string"}, "description": "List of anatomical structures to segment e.g. ['liver', 'spleen']"},
                            "modality": {"type": "string", "enum": ["CT", "MRI", "PET", "Ultrasound", "Microscopy"], "description": "Image modality (default: CT)"},
                            "window": {"type": "string", "enum": ["soft_tissue", "lung", "brain", "bone"], "description": "CT windowing preset (only for CT, default: soft_tissue)"},
                        },
                        "required": ["prompts"],
                    },
                },
                "required": ["input_path", "output_path", "params"],
            },
        }]
    }


@app.post("/mcp/tools/call")
def mcp_tools_call(req: MCPCallRequest):
    """MCP-compatible wrapper that delegates to /infer."""
    try:
        result = infer(InferRequest(**req.input))
        return {"isError": False, "content": [{"type": "text", "text": result.model_dump_json()}]}
    except Exception as e:
        return {"isError": True, "content": [{"type": "text", "text": str(e)}]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8013)
