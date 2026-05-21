# =============================================================================
# services/llm/server.py
#
# FastAPI service wrapping a Qwen causal language model.
# Used by the orchestrator's PlannerAgent, CriticAgent, ExplainerAgent, and
# ReporterAgent via LLMClient.generate_json / LLMClient.generate_text.
#
# Endpoints
# ---------
#   GET  /health          — liveness check, returns model id + GPU info
#   POST /generate_json   — generate text (expected to be a JSON object)
#   POST /generate_text   — generate plain text
#
# Environment variables
# ---------------------
#   QWEN_MODEL_ID    — HuggingFace model id (default: Qwen/Qwen2.5-3B-Instruct)
#   LLM_TEMPERATURE  — default generation temperature (default: 0.1)
#   PORT             — server port (default: 8002)
#
# GPU notes
# ---------
#   1 GPU  → Qwen2.5-3B-Instruct  (safe, fast, fits on V100-32GB)
#   2 GPUs → Qwen2.5-14B-Instruct  (better reasoning, device_map=auto splits across GPUs)
# =============================================================================

import os
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")

# float16 on GPU — 14B in fp16 ≈ 28 GB, fits across 2x V100-32GB with device_map=auto
import subprocess; subprocess.run(["nvidia-smi"])
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = int(os.getenv("PORT", "8002"))

# Effective temperature floor — 0.0 is fully greedy and can cause repetition loops;
# 0.1 gives slight diversity while remaining near-deterministic.
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

app = FastAPI(title="agentic-seg LLM (Qwen)")

tokenizer = None
model = None


def _load():
    """Load tokenizer and model on first call (lazy, also called at startup)."""
    global tokenizer, model
    if tokenizer is None:
        print(f"[LLM] Loading tokenizer: {MODEL_ID}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if model is None:
        print(f"[LLM] Loading model: {MODEL_ID} | dtype={DTYPE} | device={DEVICE}", flush=True)
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"[LLM] GPUs available: {n_gpus}", flush=True)
            for i in range(n_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory // 1024**3} GB)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print("[LLM] Model loaded and ready.", flush=True)


@app.on_event("startup")
async def startup():
    _load()


@app.get("/health")
def health():
    return {
        "ok": model is not None,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def _generate(system: str, user: str, max_new_tokens: int, temperature: float) -> str:
    """Run a single system+user turn through the model and return the decoded text."""
    _load()

    # If caller passes 0.0 (greedy), override with default to avoid repetition loops
    effective_temp = temperature if temperature > 0.0 else DEFAULT_TEMPERATURE

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=True,            # always sample; temperature controls diversity
        temperature=float(effective_temp),
        repetition_penalty=1.05,   # mild penalty to prevent repetition loops
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(
        out[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()

    return text


class GenReq(BaseModel):
    system: str
    user: str
    max_new_tokens: int = 512
    temperature: float = 0.1


@app.post("/generate_json")
def generate_json(req: GenReq):
    """Generate a response expected to be a JSON object (no schema enforcement)."""
    text = _generate(req.system, req.user, req.max_new_tokens, req.temperature)
    return {"text": text, "model_id": MODEL_ID, "device": DEVICE}


@app.post("/generate_text")
def generate_text(req: GenReq):
    """Generate a plain-text response."""
    text = _generate(req.system, req.user, req.max_new_tokens, req.temperature)
    return {"text": text, "model_id": MODEL_ID, "device": DEVICE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
