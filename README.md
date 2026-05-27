# Agentic Medical Image Segmentation

**ELEC-E8004 Project Work 2026** — Aalto University | Project #26P26

An agentic pipeline that accepts a 3D medical image (NIfTI) and a plain-language
clinical instruction, automatically selects the best segmentation tool, executes
it, and returns a verified segmentation mask with a full reasoning trace.

---

## What is this?

Three state-of-the-art segmentation tools — **TotalSegmentator**, **VoxTell**,
and **BiomedParse v2** — are deployed as containerised FastAPI microservices. A
dual-model reasoning chain (**Hulu-Med-4B** as VLM + **Qwen 2.5** as
Planner/Critic) decides which tool to use for each case. The user never needs to
know which tool to call or how to configure it.

Evaluated on **304 CT cases** (MSD benchmark):

| Condition             | Mean Dice | Successful cases |
|-----------------------|-----------|------------------|
| TotalSegmentator only | 0.4385    | 141 / 304        |
| VoxTell only          | 0.6934    | 270 / 304        |
| BiomedParse only      | 0.5538    | 234 / 304        |
| **Agentic pipeline**  | **0.8057**| **292 / 304**    |

Tool selection accuracy: **96.1 %** (292/304 cases correctly routed).

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
│   └── requirements.txt
│
├── services/
│   ├── vlm/                    # Hulu-Med-4B VLM service (port 8001)
│   ├── llm/                    # Qwen 2.5 LLM service (port 8002)
│   ├── tool_totalseg/          # TotalSegmentator service (port 8011)
│   ├── tool_voxtell/           # VoxTell service (port 8012)
│   └── tool_biomedparse/       # BiomedParse v2 service (port 8013)
│
├── benchmark_jobs/             # Per-tool SLURM evaluation scripts
│   ├── compare_results.py
│   ├── val_totalseg_job.sh
│   ├── val_voxtell_job.sh
│   └── val_biomedparse_job.sh
│
├── download_msd_ct.py          # Download MSD dataset + build cases.csv
├── validate_pipeline.py        # Run evaluation on cases.csv
├── visualize_cases.py          # Generate segmentation overlay figures
├── run_puhti.sh                # Start all services on Puhti (Apptainer)
├── ui_job.sh                   # SLURM job: launch the Gradio UI session
├── val_job.sh                  # SLURM job: full agentic pipeline validation
└── docker-compose.yml          # Local Docker Compose (for testing without GPU)
```

---

## Tool routing logic

| Modality                          | Target type                     | Tool              |
|-----------------------------------|---------------------------------|-------------------|
| CT                                | Anatomical structure (in 117-list) | TotalSegmentator |
| CT                                | Tumour / lesion / pathology     | VoxTell           |
| CT                                | Niche anatomy (not in 117-list) | VoxTell           |
| MRI                               | Any target                      | VoxTell           |
| X-Ray / Ultrasound / PET / Pathology | Any target                  | BiomedParse       |

---

## Container images (SIF files)

All Apptainer/Singularity container images are hosted on HuggingFace:

> **https://huggingface.co/datasets/csotbal/agentic-seg**

`run_puhti.sh` downloads them automatically on first run. Sizes:

| Container              | Size      |
|------------------------|-----------|
| biomedparse.sif        | 17.1 GB   |
| voxtell.sif            | 11.4 GB   |
| totalsegmentator.sif   | 9.0 GB    |
| vlm.sif                | 8.5 GB    |
| llm.sif                | 8.2 GB    |
| orchestrator.sif       | 0.2 GB    |
| **Total**              | **~55 GB**|

### Disk space requirements

| Component              | Size      |
|------------------------|-----------|
| SIF container images   | ~55 GB    |
| MSD validation dataset | ~15 GB    |
| Model caches + scratch | ~10 GB    |
| **Total needed**       | **~80 GB**|

Make sure your `$SCRATCH` partition has at least **80 GB** of free space before
starting.

---

## Running on Puhti (production)

### Prerequisites

- Access to a CSC Puhti allocation (or any HPC cluster with Apptainer)
- Python 3.10+
- Git

### 1. Clone and set up the environment

```bash
git clone https://github.com/Feanor1021/deneme.git agentic-seg
cd agentic-seg

python3 -m venv agentic
source agentic/bin/activate

pip install -r orchestrator/requirements.txt
pip install huggingface_hub
```

### 2. Create `.env`

Create a file named `.env` in the project root. Only change `SCRATCH` and
`CSC_PROJECT` to match your Puhti account — everything else can stay as-is:

```env
HF_SIF_REPO=csotbal/agentic-seg

SCRATCH=/scratch/project_XXXXXXX/your-username
CSC_PROJECT=project_XXXXXXX

VAL_DATA_DIR=/scratch/project_XXXXXXX/your-username/val_data

VLM_PORT=8001
LLM_PORT=8002
GRADIO_PORT=7860
TOTALSEG_PORT=8011
VOXTELL_PORT=8012
BIOMEDPARSE_PORT=8013
```

### 3. Start all services

```bash
cd /path/to/agentic-seg
source agentic/bin/activate
bash run_puhti.sh
```

`run_puhti.sh` does the following automatically:

1. Reads `.env` and validates required variables
2. Downloads any missing SIF files from HuggingFace
3. Kills any stale processes on the service ports
4. Starts all services as background Apptainer processes
5. Writes PIDs to `logs/<service>.pid` and logs to `logs/<service>.log`

### Wait for services to come up

Services take time to load model weights. Poll health endpoints:

```bash
for port in 8001 8002 8011 8012 8013; do
    echo -n "Port $port: "
    curl -s http://127.0.0.1:$port/health \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK' if d.get('ok') else 'NOT READY')" \
      2>/dev/null || echo "NOT READY"
done
```

Typical startup times:

| Service                           | Time       |
|-----------------------------------|------------|
| TotalSegmentator, VoxTell, BiomedParse | ~30–60 s |
| LLM (Qwen 2.5)                   | ~60–90 s   |
| VLM (Hulu-Med-4B)                | ~90–120 s  |

### Access the Gradio UI

**Option A — Interactive session:** forward port 7860 from Puhti:

```bash
ssh -L 7860:localhost:7860 <your-puhti-username>@puhti.csc.fi
```

**Option B — SLURM batch job:**

```bash
sbatch ui_job.sh
```

Check the log for the SSH tunnel command:

```bash
tail -f logs/ui_job_<JOBID>.out
```

Then open **http://localhost:7860** in your browser.

### Usage

1. Upload a NIfTI file (`.nii` or `.nii.gz`)
2. Click **Load** — a preview slice appears
3. Type your instruction, e.g. *Segment liver tumor* or *Segment the spleen*
4. Click **Run** — the pipeline runs VLM → Planner → Critic → Tool → Verifier → Reporter
5. Select a mask from the dropdown to overlay it on the CT slice
6. Expand **Plan JSON** and **Trace JSON** panels to inspect the full reasoning trace

If the Critic is unsure, a clarification row appears with **Proceed anyway** and
**Cancel** buttons.

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
source agentic/bin/activate
python download_msd_ct.py --output_dir ${VAL_DATA_DIR}
```

This downloads three MSD tasks from AWS S3 and builds `cases.csv`:

| MSD Task     | Structure          | Cases | Expected tool    |
|--------------|--------------------|-------|------------------|
| Task03 Liver | Liver organ (l=1)  | 100   | TotalSegmentator |
| Task03 Liver | Liver tumour (l=2) | 100   | VoxTell          |
| Task09 Spleen| Spleen (l=1)       | 41    | TotalSegmentator |
| Task06 Lung  | Lung tumour (l=1)  | 63    | VoxTell          |
|              | **Total**          | **304** |                |

Download time: ~20–40 minutes. Disk usage: ~15 GB.

To skip downloading and only rebuild the CSV:

```bash
python download_msd_ct.py --output_dir ${VAL_DATA_DIR} --skip_download
```

---

## Evaluation

### Full agentic pipeline (SLURM)

```bash
sbatch val_job.sh
```

### Per-tool isolated evaluation (SLURM)

```bash
sbatch benchmark_jobs/val_totalseg_job.sh
sbatch benchmark_jobs/val_voxtell_job.sh
sbatch benchmark_jobs/val_biomedparse_job.sh
```

### Compare results

```bash
python benchmark_jobs/compare_results.py
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
    --dataset_csv ${VAL_DATA_DIR}/cases.csv \
    --output_dir  ${VAL_DATA_DIR}/results \
    --force_tool  totalseg   # omit for full agentic mode
```

---

## Results

Evaluated on 304 CT cases (Task03 Liver, Task09 Spleen, Task06 Lung) from the
Medical Segmentation Decathlon.

### Overall

| Condition          | Mean Dice  | Successful  | Tool selection accuracy |
|--------------------|------------|-------------|------------------------|
| TotalSegmentator   | 0.4385     | 141 / 304   | —                      |
| VoxTell            | 0.6934     | 270 / 304   | —                      |
| BiomedParse        | 0.5538     | 234 / 304   | —                      |
| **Agentic pipeline** | **0.8057** | **292 / 304** | **96.1 %**           |

### Per structure

| Structure     | TotalSeg | VoxTell | BiomedParse | Agentic |
|---------------|----------|---------|-------------|---------|
| Liver organ   | 0.9429   | 0.5581  | 0.9451      | 0.9429  |
| Liver tumour  | 0.0000   | 0.6542  | 0.3277      | 0.6106  |
| Lung tumour   | 0.0000   | 0.8023  | 0.0277      | 0.8023  |
| Spleen        | 0.9519   | 0.9517  | 0.9592      | 0.9519  |
| **Overall**   | 0.4385   | 0.6934  | 0.5538      | **0.8057** |

The agentic pipeline matches the best-performing tool for each structure type.
The 12 misrouted cases are all liver tumour cases with short/ambiguous
instructions (e.g., "segment liver") that the VLM classified as anatomical
rather than pathological.

---

## Running locally with Docker (no GPU)

```bash
docker compose up
```

Services start on the same ports. Note that without GPU the VLM and LLM will be
very slow.

---

## Service ports reference

| Service            | Port |
|--------------------|------|
| VLM — Hulu-Med-4B | 8001 |
| LLM — Qwen 2.5    | 8002 |
| TotalSegmentator   | 8011 |
| VoxTell            | 8012 |
| BiomedParse v2     | 8013 |
| Gradio UI          | 7860 |

---

## Environment

- Python 3.10+
- PyTorch 2.10.0
- CUDA 11.8 (Puhti V100 32 GB nodes)
- CSC Puhti project allocation (set in `.env`)
- Apptainer (Singularity) for container runtime

See `orchestrator/requirements.txt` and `services/<service>/requirements.txt`
for per-component dependencies.

---

## Team

**Furkan Yardımcı** · **Haris Khan** · **Csóti Balázs Gábor** · **Leskelä Otso**

Instructor: Jiancheng Yang | Aalto University, ELEC-E8004 Project Work 2026
