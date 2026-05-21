# =============================================================================
# orchestrator/clients.py
#
# Thin HTTP client wrappers for all external microservices the orchestrator
# talks to.  Every class simply builds the correct JSON payload, POSTs it,
# and returns the parsed response dict — no business logic lives here.
#
# Classes
# -------
#   VLMClient   — Hulu VLM service  (image understanding / scene analysis)
#   LLMClient   — Qwen LLM service  (Planner / Critic / Explainer / Reporter)
#   ToolClient  — Generic segmentation tool service (TotalSeg / VoxTell / BiomedParse)
#
# Environment variable
# --------------------
#   AGSEG_HTTP_TIMEOUT  — default request timeout in seconds (default: 120)
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, Optional
import os
import time
import requests

DEFAULT_TIMEOUT = float(os.getenv("AGSEG_HTTP_TIMEOUT", "120"))


def _post_json(url: str, payload: Dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """POST *payload* as JSON to *url* and return the parsed response dict.

    Adds a '_latency_s' key to the response with the round-trip time in seconds.
    Raises requests.HTTPError on non-2xx status codes.
    """
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    out = r.json()
    if isinstance(out, dict):
        out["_latency_s"] = float(dt)
    return out


class VLMClient:
    """Hulu VLM service wrapper.

    Sends a 3-D NIfTI volume (base64-encoded) together with a free-text
    instruction and returns the model's structured analysis of the image.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def infer_3d(
        self,
        nifti_b64: str,
        axis: int,
        num_slices: int,
        instruction: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        use_think: bool = False,
    ) -> Dict[str, Any]:
        """Call the VLM /infer endpoint with a base64-encoded NIfTI volume."""
        url = f"{self.base_url}/infer"
        payload = {
            "input_type": "3d",
            "nifti_b64": nifti_b64,
            "nii_axis": int(axis),
            "nii_num_slices": int(num_slices),
            "instruction": str(instruction),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "use_think": bool(use_think),
        }
        return _post_json(url, payload)


class LLMClient:
    """Qwen (single service) wrapper.

    Exposes two generation modes used by the agent classes:
      - generate_json : expects the model to return a JSON object
      - generate_text : expects plain natural-language text
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate_json(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Call /generate_json; model is instructed to respond with a JSON object."""
        url = f"{self.base_url}/generate_json"
        payload = {
            "system": system,
            "user": user,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }
        return _post_json(url, payload)

    def generate_text(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Call /generate_text; model is instructed to respond with plain text."""
        url = f"{self.base_url}/generate_text"
        payload = {
            "system": system,
            "user": user,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }
        return _post_json(url, payload)


class ToolClient:
    """Tool service wrapper (stub now; will be real in Phase-2).

    Each segmentation tool service exposes a unified POST /<tool>/run endpoint.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def run(self, tool: str, payload: Dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
        """Call /<tool>/run with the given payload dict."""
        url = f"{self.base_url}/{tool}/run"
        return _post_json(url, payload, timeout=timeout)