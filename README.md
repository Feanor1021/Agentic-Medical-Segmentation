# Agentic Medical Image Segmentation

**ELEC-E8004 Project Work 2026 — Aalto University | Project #26P26**

An agentic pipeline that accepts a 3D medical image (NIfTI) and a plain-language clinical instruction, automatically selects the best segmentation tool, executes it, and returns a verified segmentation mask with a full reasoning trace.

---

## What is this?

Three state-of-the-art segmentation tools — TotalSegmentator, VoxTell, and BiomedParse v2 — are deployed as containerized FastAPI microservices. A dual-model reasoning chain (Hulu-Med-4B as VLM + Qwen 2.5 as Planner/Critic) decides which tool to use for each case. The user never needs to know which tool to call or how to configure it.

**Evaluated on 304 CT cases (MSD benchmark):**

| Condition | Mean Dice | Successful cases |
|---|---|---|
| TotalSegmentator only | 0.4385 | 141 / 304 |
| VoxTell only | 0.6934 | 270 / 304 |
| BiomedParse only | 0.5538 | 234 / 304 |
| **Agentic pipeline** | **0.8057** | **292 / 304** |

Tool selection accuracy: **96.1%** (292/304 cases correctly routed).

---

## Repository structure

```
.
├── orchestrator/               # Main pipeline (Gradio UI + agents)
│   ├── app.py                  # Entry point — Gradio UI + full pipeline logic
│   ├── agents.py               # PlannerAgent, CriticAgent, ExplainerAgent, ReporterAgent, VerifierAgent
│   ├── clients.py              # HTTP wrappers for VLM, LLM, and tool services
│   ├── contracts.py            # Shared dataclasses (Evidence, Plan, ToolDecision, Trace...)
│   ├── tool_registry.py        # Tool capability cards + TotalSegmentator structure list
│   └── requirements.txt        # Orchestrator Python dependencies
│
├── services/
│   ├── vlm/                    # Hulu-Med-4B VLM service (port 8001)
│   │   ├── api.py
│   │   ├── vlm.Dockerfile
│   │   ├── vlm.def             # Apptainer/Singularity definition
│   │   └── requirements.txt
│   ├── llm/                    # Qwen 2.5 LLM service (port 8002)
│   │   ├── server.py
│   │   ├── llm.Dockerfile
│   │   ├── llm.def
│   │   └── requirements.txt
│   ├── tool_totalseg/          # TotalSegmentator service (port 8011)
│   │   ├── api.py
│   │   ├── tool_totalseg.Dockerfile
│   │   └── tool_totalseg.def
│   ├── tool_voxtell/           # VoxTell service (port 8012)
│   │   ├── api.py
│   │   ├── tool_voxtell.Dockerfile
│   │   └── tool_voxtell.def
│   └── tool_biomedparse/       # BiomedParse v2 service (port 8013)
│       ├── api.py
│       ├── tool_biomedparse.Dockerfile
│       └── tool_biomedparse.def
│
├── validation/
│   ├── download_msd_ct.py      # Download MSD dataset + build cases.csv
│   ├── validate_pipeline.py    # Run evaluation on cases.csv
│   ├── compare_results.py      # Print comparison table across tools
│   └── visualize_cases.py      # Generate segmentation overlay figures
│
├── run_puhti.sh                # Start all services on Puhti (Apptainer)
├── val_job.sh                  # SLURM job: full agentic pipeline validation
├── val_totalseg_job.sh         # SLURM job: force_tool=totalseg
├── val_voxtell_job.sh          # SLURM job: force_tool=voxtell
├── val_biomedparse_job.sh      # SLURM job: force_tool=biomedparse
└── docker-compose.yml          # Local Docker Compose (no GPU required for testing)
```

---

## Tool routing logic

| Modality | Target type | Tool |
|---|---|---|
| CT | Anatomical structure (in 117-list) | TotalSegmentator |
| CT | Tumour / lesion / pathology | VoxTell |
| CT | Niche anatomy (not in 117-list) | VoxTell |
| MRI | Any target | VoxTell |
| X-Ray / Ultrasound / PET / Pathology | Any target | BiomedParse |

---

## Running on Puhti (production)

### Prerequisites

- Access to CSC Puhti under project `project_2016517`
- Singularity SIF files must exist under `apptainer/sif/`:
  - `vlm.sif`, `llm.sif`, `orchestrator.sif` — built from `.def` files
  - `totalsegmentator.sif` — available at `/scratch/project_2016517/haris/totalsegmentator.sif`
  - `voxtell.sif` — built from `tool_voxtell.def`
  - `biomedparse.sif` — available at `/scratch/project_2016517/balazs/biomedparse/biomedparse.sif`

### Start all services

```bash
cd /scratch/project_2016517/furkan/agentic-seg
source agentic/bin/activate
bash run_puhti.sh
```

`run_puhti.sh` does the following automatically:
1. Kills any stale processes on ports 8001, 8002, 8011, 8012, 8013, 7860
2. Builds any missing SIF files (`vlm.sif`, `llm.sif`, `orchestrator.sif`)
3. Starts all services as background Apptainer processes
4. Writes PIDs to `logs/<service>.pid` and logs to `logs/<service>.log`

### Wait for services to come up

Services take time to load model weights. Poll health endpoints:

```bash
# Check all services (run after ~60–90s)
for port in 8001 8002 8011 8012 8013; do
    echo -n "Port $port: "
    curl -s http://127.0.0.1:$port/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK' if d.get('ok') else 'NOT READY')" 2>/dev/null || echo "NOT READY"
done
```

Typical startup times:
- TotalSegmentator, VoxTell, BiomedParse: **~30–60 seconds**
- LLM (Qwen 2.5): **~60–90 seconds**
- VLM (Hulu-Med-4B): **~90–120 seconds**

### Access the Gradio UI

Once all services are up, forward port 7860 from Puhti to your local machine:

```bash
# From your local terminal
ssh -L 7860:localhost:7860 <your-puhti-username>@puhti.csc.fi
```

Then open **http://localhost:7860** in your browser.

**Usage:**
1. Upload a NIfTI file (`.nii` or `.nii.gz`)
2. Click **Load** — a preview slice appears
3. Type your instruction, e.g. `Segment liver tumor` or `Segment the spleen`
4. Click **Run** — the pipeline runs VLM → Planner → Critic → Tool → Verifier → Reporter
5. Select a mask from the dropdown to overlay it on the CT slice
6. Expand **Plan JSON** and **Trace JSON** panels to inspect the full reasoning trace

If the Critic is unsure, a clarification row appears with **Proceed anyway** and **Cancel** buttons.

### Stop all services

```bash
for svc in vlm llm totalseg voxtell biomedparse orchestrator; do
    pid=$(cat logs/${svc}.pid 2>/dev/null)
    [ -n "$pid" ] && kill $pid 2>/dev/null
done
```

---

## Dataset

### Download

```bash
cd /scratch/project_2016517/furkan/agentic-seg
source agentic/bin/activate
python download_msd_ct.py --output_dir /scratch/project_2016517/furkan/val_data
```

This downloads three MSD tasks from AWS S3 and builds `cases.csv`:

| MSD Task | Structure | Cases | Expected tool |
|---|---|---|---|
| Task03 Liver | Liver organ (label=1) | 100 | TotalSegmentator |
| Task03 Liver | Liver tumour (label=2) | 100 | VoxTell |
| Task09 Spleen | Spleen (label=1) | 41 | TotalSegmentator |
| Task06 Lung | Lung tumour (label=1) | 63 | VoxTell |
| **Total** | | **304** | |

Download time: ~20–40 minutes depending on network. Disk usage: ~15 GB.

To skip downloading and only rebuild the CSV from existing files:

```bash
python download_msd_ct.py \
    --output_dir /scratch/project_2016517/furkan/val_data \
    --skip_download
```

To validate an existing CSV without downloading or generating:

```bash
python download_msd_ct.py \
    --output_dir /scratch/project_2016517/furkan/val_data \
    --validate_only
```

---

## Evaluation

### Option A — Full agentic pipeline (SLURM)

```bash
sbatch val_job.sh
```

Starts all services, waits for health checks, then runs `validate_pipeline.py` with the full VLM → Planner → Critic routing. Results saved to `/scratch/project_2016517/furkan/val_data/results/`.

### Option B — Per-tool isolated evaluation (SLURM)

```bash
sbatch val_totalseg_job.sh      # force_tool=totalseg
sbatch val_voxtell_job.sh       # force_tool=voxtell
sbatch val_biomedparse_job.sh   # force_tool=biomedparse
```

Each job starts services, waits for the relevant tool's health check, then runs validation with that tool forced for all cases. Results saved to `results_totalseg/`, `results_voxtell/`, `results_biomedparse/`.

### Compare results

After all four jobs complete:

```bash
python compare_results.py
```

Example output:

```
============================================================
Tool            Cases    Success   MeanDice   Median
------------------------------------------------------------
totalseg          304  141/304     0.4385     0.9453
voxtell           304  270/304     0.6934     0.8023
biomedparse       304  234/304     0.5538     0.9451
agentic           304  292/304     0.8057     0.9429
============================================================
```

### Manual run (no SLURM)

```bash
python validate_pipeline.py \
    --dataset_csv /scratch/project_2016517/furkan/val_data/cases.csv \
    --output_dir  /scratch/project_2016517/furkan/val_data/results \
    --force_tool  totalseg   # omit for full agentic mode
```

---

## Results

Evaluated on 304 CT cases (Task03 Liver, Task09 Spleen, Task06 Lung) from the Medical Segmentation Decathlon.

### Overall

| Condition | Mean Dice | Successful | Tool selection accuracy |
|---|---|---|---|
| TotalSegmentator | 0.4385 | 141 / 304 | — |
| VoxTell | 0.6934 | 270 / 304 | — |
| BiomedParse | 0.5538 | 234 / 304 | — |
| **Agentic pipeline** | **0.8057** | **292 / 304** | **96.1%** |

### Per structure

| Structure | TotalSeg | VoxTell | BiomedParse | Agentic |
|---|---|---|---|---|
| Liver organ | 0.9429 | 0.5581 | 0.9451 | **0.9429** |
| Liver tumour | 0.0000 | 0.6542 | 0.3277 | **0.6106** |
| Lung tumour | 0.0000 | 0.8023 | 0.0277 | **0.8023** |
| Spleen | 0.9519 | 0.9517 | 0.9592 | **0.9519** |
| **Overall** | 0.4385 | 0.6934 | 0.5538 | **0.8057** |

The agentic pipeline matches the best-performing tool for each structure type. The 12 misrouted cases are all liver tumour cases with short/ambiguous instructions (e.g., "segment liver") that the VLM classified as anatomical rather than pathological.

---

## Running locally with Docker (no GPU)

For quick testing without Puhti:

```bash
docker compose up
```

Services start on the same ports (8001, 8002, 8011, 8012, 8013, 7860). Note that without GPU the VLM and LLM will be very slow; tool services that require model weights (VoxTell, BiomedParse) need the weights present at the expected paths.

---

## Service ports reference

| Service | Port |
|---|---|
| VLM — Hulu-Med-4B | 8001 |
| LLM — Qwen 2.5 | 8002 |
| TotalSegmentator | 8011 |
| VoxTell | 8012 |
| BiomedParse v2 | 8013 |
| Gradio UI | 7860 |

---

## Environment

- Python 3.10+
- PyTorch 2.10.0
- CUDA 11.8 (Puhti V100 32 GB nodes)
- CSC Puhti project allocation: `project_2016517`
- Apptainer (Singularity) for container runtime

See `orchestrator/requirements.txt` and `services/<service>/requirements.txt` for per-component dependencies.

---

## Team

Furkan Yardımcı · Haris Khan · Csóti Balázs Gábor · Leskelä Otso

Instructor: Jiancheng Yang | Aalto University, ELEC-E8004 Project Work 2026#   d e n e m e  
 