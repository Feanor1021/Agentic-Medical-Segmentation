# =============================================================================
# orchestrator/clients2.py
#
# Alternative HTTP client wrappers — an earlier API contract revision.
# Kept alongside clients.py for reference / migration purposes.
#
# Differences from clients.py
# ---------------------------
#   VLMClient.infer_3d  — accepts a file path instead of base64 data and calls
#                         /infer_3d rather than /infer.
#   LLMClient           — single .reason(mode, context) call instead of
#                         separate generate_json / generate_text endpoints.
#   ToolClient          — exposes .segment() calling POST /segment instead of
#                         POST /<tool>/run.
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

    Adds a '_latency_s' key with the round-trip time in seconds.
    Raises requests.HTTPError on non-2xx status codes.
    """
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    out = r.json()
    if isinstance(out, dict):
        out["_latency_s"] = dt
    return out


class VLMClient:
    """Hulu VLM service wrapper (path-based variant).

    Unlike the base64 variant in clients.py, this version passes the NIfTI
    file path directly and calls the /infer_3d endpoint.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def infer_3d(
        self,
        nifti_path: str,
        prompt: str,
        axis: int = 2,
        num_slices: int = 160,
    ) -> Dict[str, Any]:
        """Call /infer_3d with a server-accessible NIfTI file path."""
        url = f"{self.base_url}/infer_3d"
        payload = {
            "nifti_path": nifti_path,
            "prompt": prompt,
            "axis": axis,
            "num_slices": num_slices,
        }
        return _post_json(url, payload)


class LLMClient:
    """Qwen reasoning service wrapper (unified endpoint variant).

    Uses a single /reason endpoint with a mode string rather than the
    separate generate_json / generate_text split in clients.py.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reason(self, mode: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call /reason with the given mode and context dict."""
        url = f"{self.base_url}/reason"
        payload = {"mode": mode, "context": context}
        return _post_json(url, payload)


class ToolClient:
    """Generic tool wrapper (unified /segment endpoint variant).

    Each tool exposes POST /segment; the tool identity is encoded inside
    the payload rather than in the URL path.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def segment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call /segment with the given payload dict."""
        url = f"{self.base_url}/segment"
        return _post_json(url, payload)