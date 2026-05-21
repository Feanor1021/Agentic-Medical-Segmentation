# =============================================================================
# services/tool_totalseg/api.py
#
# FastAPI wrapper around the TotalSegmentator CLI.
# Accepts a NIfTI input path + optional structure list, runs TotalSegmentator,
# and returns the output directory path containing per-structure .nii.gz masks.
#
# Endpoints
# ---------
#   GET  /health          — liveness check
#   POST /infer           — run segmentation (InferRequest → dict)
#   GET  /mcp/tools/list  — MCP tool manifest
#   POST /mcp/tools/call  — MCP tool invocation wrapper
#
# Params (inside InferRequest.params)
# ------------------------------------
#   structure   — single structure name (string), normalised to structures list
#   structures  — list of structure names; empty = full segmentation (all 117)
# =============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import subprocess

app = FastAPI(title="TotalSegmentator API", version="1.0")

# All 117 anatomical structures supported by TotalSegmentator.
# Requests referencing names not in this list are rejected with 400.
SUPPORTED_STRUCTURES = [
    'adrenal_gland_left', 'adrenal_gland_right', 'aorta', 'atrial_appendage_left',
    'autochthon_left', 'autochthon_right', 'brachiocephalic_trunk', 'brachiocephalic_vein_left',
    'brachiocephalic_vein_right', 'brain', 'clavicula_left', 'clavicula_right', 'colon',
    'common_carotid_artery_left', 'common_carotid_artery_right', 'costal_cartilages',
    'duodenum', 'esophagus', 'femur_left', 'femur_right', 'gallbladder',
    'gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left', 'gluteus_medius_right',
    'gluteus_minimus_left', 'gluteus_minimus_right', 'heart', 'hip_left', 'hip_right',
    'humerus_left', 'humerus_right', 'iliac_artery_left', 'iliac_artery_right',
    'iliac_vena_left', 'iliac_vena_right', 'iliopsoas_left', 'iliopsoas_right',
    'inferior_vena_cava', 'kidney_cyst_left', 'kidney_cyst_right', 'kidney_left', 'kidney_right',
    'liver', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right',
    'lung_upper_lobe_left', 'lung_upper_lobe_right', 'pancreas', 'portal_vein_and_splenic_vein',
    'prostate', 'pulmonary_vein', 'rib_left_1', 'rib_left_10', 'rib_left_11', 'rib_left_12',
    'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7',
    'rib_left_8', 'rib_left_9', 'rib_right_1', 'rib_right_10', 'rib_right_11', 'rib_right_12',
    'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6', 'rib_right_7',
    'rib_right_8', 'rib_right_9', 'sacrum', 'scapula_left', 'scapula_right', 'skull',
    'small_bowel', 'spinal_cord', 'spleen', 'sternum', 'stomach', 'subclavian_artery_left',
    'subclavian_artery_right', 'superior_vena_cava', 'thyroid_gland', 'trachea',
    'urinary_bladder', 'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4',
    'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7', 'vertebrae_L1', 'vertebrae_L2',
    'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5', 'vertebrae_S1', 'vertebrae_T1',
    'vertebrae_T10', 'vertebrae_T11', 'vertebrae_T12', 'vertebrae_T2', 'vertebrae_T3',
    'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8', 'vertebrae_T9',
]


class InferRequest(BaseModel):
    input_path: str
    output_path: str
    params: dict = {}


class MCPCallRequest(BaseModel):
    name: str
    input: Dict[str, Any]


@app.get("/health")
def health():
    return {"ok": True, "tool": "TotalSegmentator", "version": "2.12.0"}


@app.post("/infer")
def infer(request: InferRequest):
    """Run TotalSegmentator on the given NIfTI file.

    If params.structures is provided, passes --roi_subset to limit output to
    those structures (faster).  Empty list = full segmentation (all 117).
    """
    structure = request.params.get("structure")
    structures = request.params.get("structures", [])
    if structure and not structures:
        structures = [structure]

    if not request.input_path.endswith((".nii.gz", ".nii")):
        raise HTTPException(status_code=400, detail="Only .nii or .nii.gz accepted.")

    invalid = [s for s in structures if s not in SUPPORTED_STRUCTURES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Structures not supported: {invalid}")

    cmd = ["TotalSegmentator", "-i", request.input_path, "-o", request.output_path]
    if structures:
        cmd += ["--roi_subset"] + structures

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return {
            "ok": True,
            "mask_path": request.output_path,
            "structures": structures,
            "message": f"Segmented: {structures if structures else 'all'}",
        }
    else:
        raise HTTPException(status_code=500, detail=result.stderr)


@app.get("/mcp/tools/list")
def mcp_tools_list():
    return {
        "tools": [{
            "name": "totalseg",
            "description": "CT organ segmentation, 117 structures.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["input_path", "output_path"],
            },
        }]
    }


@app.post("/mcp/tools/call")
def mcp_tools_call(req: MCPCallRequest):
    """MCP-compatible wrapper that delegates to /infer."""
    try:
        result = infer(InferRequest(**req.input))
        return {"isError": False, "content": [{"type": "text", "text": str(result)}]}
    except Exception as e:
        return {"isError": True, "content": [{"type": "text", "text": str(e)}]}
