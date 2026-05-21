# =============================================================================
# services/vlm/api.py
#
# FastAPI service wrapping the Hulu-Med Vision-Language Model.
# Accepts a base64-encoded NIfTI volume and a free-text instruction,
# runs the VLM, and returns a structured JSON analysis of the image.
#
# The response text is consumed by the orchestrator's on_run() function
# which parses the first JSON object out of it to build the plan.
#
# Endpoints
# ---------
#   GET  /health  — liveness check
#   POST /infer   — main inference endpoint (VLMInferRequest → VLMInferResponse)
#
# Environment variables
# ---------------------
#   HULU_MODEL_ID  — HuggingFace model id (default: ZJU-AI4H/Hulu-Med-4B)
#   HOST           — bind host (default: 0.0.0.0)
#   PORT           — server port (default: 8000)
# =============================================================================

import os
import base64
import tempfile
from typing import Optional, Literal

import torch
import nibabel as nib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = os.getenv("HULU_MODEL_ID", "ZJU-AI4H/Hulu-Med-4B")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="agentic-seg VLM (Hulu-Med)")

model = None
processor = None
tokenizer = None


def build_planner_prompt(user_instruction: str) -> str:
    """Build the system prompt injected into the VLM conversation.

    The prompt instructs the VLM to act as a Planner that produces a
    CRITIC-READY DECISION CONTEXT from image evidence + user text.
    Output must be a single JSON object matching the schema defined here.
    """
    return f"""
You are the PLANNER of an agentic medical image segmentation system.
Your output is consumed by a downstream CRITIC agent for decision-making and tool routing.
This output is NOT for end users.

INPUTS:
- A 3D medical image volume (NIfTI)
- A USER instruction (may be noisy, slang, ambiguous, or contradictory)

PRIMARY OBJECTIVE:
Produce a CRITIC-READY DECISION CONTEXT grounded in IMAGE EVIDENCE first,
and USER TEXT second.

===========================
MANDATORY RULES (STRICT)
===========================

IMAGE-FIRST RESOLUTION (MANDATORY):
- If the IMAGE provides sufficient evidence to identify modality or anatomy,
  you MUST resolve ambiguity using the IMAGE.
- You are NOT allowed to defer to "need_clarification" if image evidence is sufficient.
- User confusion must NOT override clear image evidence.

CONFIDENCE RULES:
- confidence must be a float in [0.0, 1.0].
- If image evidence is clear, set confidence >= 0.7.
- NEVER set confidence to 0.0 when image evidence exists.
- Use "unknown" ONLY if the image truly does not support a decision.

STATUS RULE:
- status = "need_clarification" ONLY IF:
  * image evidence is insufficient AND
  * modality OR anatomy OR target cannot be inferred.
- Otherwise, status MUST be "ok".

INTENT SUMMARY RULE:
- intent_summary MUST reflect the FINAL resolved intent.
- Do NOT mention uncertainty if ambiguity.resolved_by_image = true.

TARGET RULE:
- target must be tool-routable and generic.
- Prefer "lung lesion" over overly specific phrasing.
- If pathology is known, include it in parentheses.
  Example: "lung lesion (ground-glass opacities)"

PLAN RULE (CRITIC-LEVEL ONLY):
- Plan steps must describe DECISION GATES and TOOL ROUTING JUSTIFICATION.
- Do NOT describe execution actions like "segment", "refine", "export".

PLAN EXAMPLE (SEMANTICS ONLY):

BAD:
- "Segment lung lesion"

GOOD:
- "Validate that a lung CT lesion-segmentation tool supports 3D ground-glass opacity segmentation"

EVIDENCE:
- Provide 2–3 bullets justifying modality, anatomy, and task intent.
- At least one bullet must explicitly cite IMAGE evidence.

QUESTIONS:
- Only ask questions if status="need_clarification".
- Questions must unblock tool routing.

===========================
OUTPUT FORMAT (STRICT)
===========================
- Output MUST be valid JSON only.
- Output MUST start with "{{" and end with "}}".
- No markdown, no extra text.

RETURN JSON WITH THIS EXACT SCHEMA:
{{
  "status": "ok" | "need_clarification",
  "user": {{
    "raw_input": {user_instruction!r},
    "intent_summary": "<FINAL resolved intent>",
    "ambiguity": {{
      "present": true | false,
      "resolved_by_image": true | false,
      "notes": "<what was ambiguous and how it was resolved>"
    }}
  }},
  "image": {{
    "modality": {{"label":"CT|MRI|PET|XR|US|unknown","confidence":0.0,"notes":""}},
    "anatomy": {{"region":"<canonical region>","confidence":0.0,"notes":""}}
  }},
  "task_intent": {{
    "type":"segmentation|diagnosis_question|localization|unknown",
    "target":"<tool-routable target>",
    "constraints":"<3D/2D, mask type, or empty>"
  }},
  "evidence": {{
    "summary":["<bullet 1>","<bullet 2>"]
  }},
  "plan": [
    {{
      "id": 1,
      "name": "<decision step>",
      "goal": "<why tool selection is safe>",
      "inputs": ["<needed info>"],
      "outputs": ["<decision artifact>"]
    }}
  ],
  "message_to_user":"<simple explanation>",
  "questions":[]
}}
""".strip()


# ----------------------------
# Request / Response contracts
# ----------------------------

class VLMInferRequest(BaseModel):
    input_type: Literal["3d"] = "3d"
    nifti_b64: str               # base64-encoded .nii or .nii.gz bytes
    nii_axis: int = 2            # 0=sagittal, 1=coronal, 2=axial
    nii_num_slices: int = 160
    instruction: str             # free-text goal, e.g. "Segment lung infection"
    max_new_tokens: int = 1024
    temperature: float = 0.2
    use_think: bool = False      # Hulu processor decode option (if supported)


class VLMInferResponse(BaseModel):
    model_id: str
    device: str
    text: str


# ----------------------------
# Model load
# ----------------------------

@app.on_event("startup")
def _load():
    """Load the Hulu-Med model and processor at service startup."""
    global model, processor, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
    }


# ----------------------------
# Inference
# ----------------------------

@app.post("/infer", response_model=VLMInferResponse)
def infer(req: VLMInferRequest):
    """Decode the NIfTI volume, build the VLM conversation, and return model output."""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Decode base64 payload
    try:
        raw = base64.b64decode(req.nifti_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    with tempfile.TemporaryDirectory() as td:
        # Detect gzip magic bytes to choose correct file extension
        is_gz = len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B
        nii_path = os.path.join(td, "input.nii.gz" if is_gz else "input.nii")

        with open(nii_path, "wb") as f:
            f.write(raw)

        # Validate that nibabel can read the file before passing to the model
        try:
            _ = nib.load(nii_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid NIfTI (is_gz={is_gz}): {e}",
            )

        planner_prompt = build_planner_prompt(req.instruction)

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": planner_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "3d",
                        "3d": {
                            "image_path": nii_path,
                            "nii_num_slices": int(req.nii_num_slices),
                            "nii_axis": int(req.nii_axis),
                        },
                    }
                ],
            },
        ]

        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)

        gen_kwargs = {
            "max_new_tokens": int(req.max_new_tokens),
            "do_sample": req.temperature > 0,
            "temperature": float(req.temperature),
        }

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Hulu processor may or may not support use_think — try with, fall back without
        try:
            text = processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                use_think=bool(req.use_think),
            )[0].strip()
        except TypeError:
            text = processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )[0].strip()

        return VLMInferResponse(
            model_id=MODEL_ID,
            device=DEVICE,
            text=text,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
