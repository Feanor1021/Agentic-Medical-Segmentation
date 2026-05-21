# =============================================================================
# services/tool_voxtell/api.py
#
# FastAPI wrapper around the VoxTell CLI (voxtell-predict).
# Accepts a NIfTI input path + text prompts, handles 4-D / multi-channel
# inputs automatically, runs VoxTell, and returns the output path.
#
# Endpoints
# ---------
#   GET  /health          — liveness check
#   POST /infer           — run segmentation (InferRequest → InferResponse)
#   GET  /mcp/tools/list  — MCP tool manifest
#   POST /mcp/tools/call  — MCP tool invocation wrapper
#
# Params (inside InferRequest.params)
# ------------------------------------
#   prompts     — comma-separated string or list of target structures
#   text_prompt — alias for prompts (either key is accepted)
#
# Environment variables
# ---------------------
#   VOXTELL_MODEL_DIR  — path to VoxTell weights (default: /app/voxtell_weights/voxtell_v1.1)
# =============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Union
import subprocess
import os
import tempfile

app = FastAPI(title="VoxTell API", version="1.0")

MODEL_DIR = os.getenv("VOXTELL_MODEL_DIR", "/app/voxtell_weights/voxtell_v1.1")


class InferRequest(BaseModel):
    input_path: str
    output_path: str
    params: dict = {}


class InferResponse(BaseModel):
    status: str
    output_path: Optional[str] = None
    message: Optional[str] = None


class MCPCallRequest(BaseModel):
    name: str
    input: Dict[str, Any]


def _ensure_single_channel(input_path: str) -> str:
    """Reduce a 4-D (multi-channel) NIfTI to a single 3-D volume.

    BraTS and similar datasets ship 4-D files with shape (X, Y, Z, C).
    VoxTell expects 3-D input, so we extract the most informative channel:
      - index 1 (T1ce) when ≥2 channels are present (best for tumors)
      - index 0 otherwise

    Returns the original path unchanged if the volume is already 3-D.
    The caller is responsible for deleting any temporary file created.
    Raises HTTPException(422) if extraction fails.
    """
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(input_path)
        vol = img.get_fdata()

        if vol.ndim <= 3:
            return input_path  # already 3-D, nothing to do

        n_channels = vol.shape[3]
        channel_idx = 1 if n_channels >= 2 else 0  # prefer T1ce (index 1)
        vol_3d = vol[..., channel_idx].astype(np.float32)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".nii.gz", delete=False,
            dir=os.path.dirname(input_path) or "/tmp",
        )
        nib.save(nib.Nifti1Image(vol_3d, img.affine, img.header), tmp.name)
        tmp.close()
        return tmp.name

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"4D input detected but channel extraction failed: {e}",
        )


@app.get("/health")
def health():
    return {"ok": True, "status": "ok", "tool": "VoxTell"}


@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest):
    """Run VoxTell segmentation.

    Resolves prompts from params, handles 4-D inputs, then calls
    voxtell-predict and cleans up any temporary files.
    """
    raw = request.params.get("prompts") or request.params.get("text_prompt") or ""
    if isinstance(raw, list):
        prompts = raw
    else:
        prompts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not prompts:
        raise HTTPException(status_code=422, detail="params.prompts required")
    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=400, detail=f"Input not found: {request.input_path}")

    os.makedirs(request.output_path, exist_ok=True)

    # Handle 4-D / multi-channel inputs
    tmp_path = None
    actual_input = request.input_path
    try:
        resolved = _ensure_single_channel(request.input_path)
        if resolved != request.input_path:
            tmp_path = resolved
            actual_input = resolved
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        cmd = [
            "voxtell-predict",
            "-i", actual_input,
            "-o", request.output_path,
            "-m", MODEL_DIR,
            "-p", *prompts,
            "--save-combined",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {result.stderr[:500]}",
            )
    finally:
        # Always clean up the temporary 3-D file if one was created
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    out_file = os.path.join(request.output_path, os.path.basename(request.input_path))
    return InferResponse(
        status="success",
        output_path=out_file,
        message=f"Segmented: {', '.join(prompts)}",
    )


@app.get("/mcp/tools/list")
def mcp_tools_list():
    return {
        "tools": [{
            "name": "voxtell",
            "description": "Instruction-based 3D segmentation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "prompts": {"type": "string", "description": "e.g. 'liver,spleen'"},
                        },
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
        return {"isError": False, "content": [{"type": "text", "text": result.json()}]}
    except Exception as e:
        return {"isError": True, "content": [{"type": "text", "text": str(e)}]}
