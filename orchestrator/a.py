# =============================================================================
# a.py  —  Quick smoke-test script for the VLM /infer endpoint.
#
# Loads a NIfTI file, base64-encodes it, and fires a single POST request to
# the VLM service to verify the endpoint is reachable and returns a response.
# Not part of the orchestrator pipeline — run manually for ad-hoc testing.
#
# Usage
# -----
#   VLM_URL=http://<host>:<port> python a.py
#
# Environment variables
# ---------------------
#   VLM_URL  — base URL of the VLM service (default: http://127.0.0.1:8000)
# =============================================================================

import base64
import json
import time
import os
import requests

VLM_URL = os.environ.get("VLM_URL", "http://127.0.0.1:8000")
nii_path = "coronacases_org_003.nii"  # adjust path if needed

with open(nii_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "input_type": "3d",
    "nifti_b64": b64,
    "nii_axis": 2,
    "nii_num_slices": 64,
    "instruction": "Describe abnormal findings briefly and suggest next segmentation target. Return JSON with keys: target, rationale.",
    "max_new_tokens": 256,
    "temperature": 0.2,
    "use_think": False,
}

t0 = time.time()
r = requests.post(f"{VLM_URL}/infer", json=payload, timeout=600)
dt = time.time() - t0

print("status:", r.status_code, "time_s:", round(dt, 2))
print(r.text[:2000])