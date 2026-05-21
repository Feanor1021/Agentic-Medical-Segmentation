import os
import subprocess
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union

app = FastAPI(title="VoxTell API", version="1.0")

class InferRequest(BaseModel):
    image_path: str
    text_prompts: Union[str, List[str]]
    model_dir: str = "/app/voxtell_weights/voxtell_v1.1"
    output_dir: str = "/data/output"

@app.on_event("startup")
async def startup_event():
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}", flush=True)
        print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("WARNING: No GPU detected — running on CPU!", flush=True)

@app.post("/infer")
def infer(request: InferRequest):
    prompts = [request.text_prompts] if isinstance(request.text_prompts, str) else request.text_prompts

    os.makedirs(request.output_dir, exist_ok=True)

    cmd = [
        "voxtell-predict",
        "-i", request.image_path,
        "-o", request.output_dir,
        "-m", request.model_dir,
        "-p", *prompts,
        "--save-combined"
    ]

    print(f"Running: {' '.join(cmd)}", flush=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {result.stderr}")

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    base_name = os.path.basename(request.image_path)
    output_path = os.path.join(request.output_dir, base_name)

    return {
        "status": "success",
        "message": f"Inference complete for {len(prompts)} prompts.",
        "output_file": output_path
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8105)
