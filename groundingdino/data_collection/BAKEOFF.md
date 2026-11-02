# Detector Bake-off (Workflow 1) — Setup Guide

**Audience:** humans or coding agents setting up / re-running the GroundingDINO vs YOLO localization bake-off.

**Goal:** On the **same close-up frames**, compare:

| Strategy | What it detects | How |
|----------|-----------------|-----|
| **GroundingDINO** | Players + logos in **one** joint pass | Zero-shot phrases: `person/player/people/man` + `logo/text` |
| **YOLOv8n** | Players and logos in **two** models | Fine-tuned `player_detector` + class-agnostic `logo_detector` |

**Out of scope (Workflow 2, not built here):** keeping logos inside player boxes, ResNet brand classification.

**Branch:** `omar-benchmark-groundingdino-yolo`  
**Code lives in:** `csc490-group2/groundingdino/data_collection/`  
**Heavy files live outside git:** workspace `local-data/groundingdino/data_collection/` (weights, videos, frames, Roboflow downloads, comparison JPGs). `*.pt` / `*.pth` are gitignored.

---

## 1. Repo vs `local-data` (critical)

Assume a workspace like:

```text
csc490/                                    ← workspace root (parent of the git clone)
  csc490-group2/                           ← git repo
    groundingdino/data_collection/         ← scripts + this doc + Colab notebook
  local-data/groundingdino/data_collection/  ← ALL runtime data + weights
```

`paths.py` walks upward until it finds `local-data/groundingdino/data_collection/`. If that folder is missing, scripts fall back to writing under the package dir (usually wrong for this project).

**Always create / populate `local-data/...` next to the clone**, not inside the git tree.

---

## 2. Where every file type goes

```text
local-data/groundingdino/data_collection/
├── weights/
│   ├── groundingdino_swint_ogc.pth          ← DINO “low” (Swin-T) — required for --model low
│   ├── groundingdino_swinb_cogcoor.pth      ← DINO “high” (Swin-B) — optional
│   └── yolo/
│       ├── player_detector/best.pt          ← trained player YOLO
│       └── logo_detector/best.pt            ← trained logo YOLO (class-agnostic)
│
└── data/
    ├── input/
    │   ├── raw_videos/
    │   │   └── athletico_real_2023.mp4      ← source match (or your own .mp4)
    │   ├── framed_videos/
    │   │   └── athletico_real_2023/
    │   │       ├── close_ups/               ← bake-off input frames (chronological names)
    │   │       └── wide_shots/
    │   └── logos/                           ← optional brand image folders (NOT used by YOLO train yet)
    │       └── riyadh_air/*.jpg
    │
    ├── output/
    │   └── comparison/                      ← bake-off side-by-side results
    │       ├── groundingdino/
    │       │   ├── combined/{annotated_frames,txt_labels}/
    │       │   ├── players/{annotated_frames,txt_labels}/
    │       │   └── logos/{annotated_frames,txt_labels}/
    │       └── yolo/
    │           ├── players/{annotated_frames,txt_labels}/
    │           └── logos/{annotated_frames,txt_labels}/
    │
    └── yolo_train/                          ← created by download/train scripts
        ├── roboflow/                        ← downloaded Roboflow YOLO exports
        │   ├── manifest.json
        │   ├── player_soccer/
        │   ├── sponsor_detector/
        │   └── logo_detection_e92p3/
        ├── merged_logo_detector/            ← 1-class merge of the two logo Roboflow sets
        └── runs/                            ← Ultralytics train runs (then copied → weights/yolo/)
```

### YOLO TXT class ids

| File set | Class 0 | Class 1 |
|----------|---------|---------|
| DINO `combined` / split labels | person | logo |
| YOLO players | player | — |
| YOLO logos | logo (agnostic; **not** brand ID) | — |

Brand names (Riyadh Air, Hyundai, …) are **not** YOLO classes. Classification is Workflow 2.

---

## 3. Prerequisites

### Python environments

Two common setups on this machine:

| Task | Env that works |
|------|----------------|
| YOLO train / YOLO labelers | Project `.venv` with `ultralytics`, `roboflow`, `opencv-python`, `torch` (MPS/CUDA OK) |
| GroundingDINO `auto_labeler.py` | Conda/`base` (or any env) where the **`groundingdino` package** is installed |

```bash
# From csc490-group2 (YOLO side)
python -m venv .venv && source .venv/bin/activate
pip install ultralytics roboflow pyyaml opencv-python torch torchvision
```

GroundingDINO install is project-specific (IDEA-Research GroundingDINO). Do **not** expect `pip install groundingdino` alone to match this repo’s `auto_labeler.py` imports (`groundingdino.util.inference`).

### Weights to obtain once

1. **DINO checkpoints** → `local-data/.../weights/`
   - Low: [GroundingDINO Swin-T OGC](https://github.com/IDEA-Research/GroundingDINO) (`groundingdino_swint_ogc.pth`)
   - High (optional): Swin-B cogcoor (`groundingdino_swinb_cogcoor.pth`)

2. **YOLO `best.pt`** → train (below) or unzip Colab export into `weights/yolo/{player_detector,logo_detector}/best.pt`

---

## 4. Extract close-up frames

```bash
cd csc490-group2/groundingdino/data_collection

# Put the mp4 at: local-data/.../data/input/raw_videos/athletico_real_2023.mp4
python frame_extractor.py \
  --video athletico_real_2023.mp4 \
  --start-minutes 0 \
  --minutes 10 \
  --interval 1
```

**Behavior (do not “improve” without asking):**

- `green_ratio > 0.40` → `wide_shots/`; else → `close_ups/` (original heuristic).
- Names are chronological: `frame_{seconds:05d}s_{rem:02d}.jpg` (video time, not extract order).
- Re-extracting the same window overwrites the same filenames.

Bake-off inputs are **`.../framed_videos/<match>/close_ups/`**.

---

## 5. Train YOLO detectors

Logo YOLO training data (class-agnostic merge) comes **only** from Roboflow:

| Local name | Roboflow | Role |
|------------|----------|------|
| `sponsor_detector` | [sponsor-detector v2](https://universe.roboflow.com/myworkspace1-mvqud/sponsor-detector/dataset/2) | Soccer-ish sponsors (Adidas, Emirates, Nike, Spotify) — better transfer to kits |
| `logo_detection_e92p3` | [logo-detection-e92p3 v4](https://universe.roboflow.com/data-deo7j/logo-detection-e92p3/dataset/4) | General logos in the wild — weak transfer to jersey logos |

Player YOLO: Roboflow `school-hnpey` / `soccer-iqjx5` v2 → `roboflow/player_soccer/`.

API key and dataset IDs: `yolo/config.py` (`ROBOFLOW_API_KEY`, `LOGO_DETECTION_SOURCES`, `PLAYER_*`).

### Option A — Colab (recommended for GPU)

1. Upload [`YOLO_Bakeoff_Train_Colab.ipynb`](YOLO_Bakeoff_Train_Colab.ipynb) to Colab.
2. Runtime → GPU → Run all.
3. Download `yolo_bakeoff_weights.zip`.
4. Unzip so you have:

```text
local-data/groundingdino/data_collection/weights/yolo/player_detector/best.pt
local-data/groundingdino/data_collection/weights/yolo/logo_detector/best.pt
```

### Option B — Local

```bash
cd csc490-group2/groundingdino/data_collection/yolo
python download_datasets.py           # → data/yolo_train/roboflow/
python build_merged_logo_dataset.py   # → data/yolo_train/merged_logo_detector/ (nc=1, names=[logo])
python train_player_yolo.py           # default 25 epochs → weights/yolo/player_detector/best.pt
python train_logo_yolo.py             # default 5 epochs  → weights/yolo/logo_detector/best.pt
```

---

## 6. Run the bake-off (inference)

Working directory: `csc490-group2/groundingdino/data_collection/`

`--directory` is relative to `framed_videos/` (e.g. `athletico_real_2023/close_ups`).

### GroundingDINO (joint)

```bash
# Use the env that has the groundingdino package. CPU is slow (~3–4s/frame).
python auto_labeler.py \
  --directory athletico_real_2023/close_ups \
  --model low \
  --device cpu \
  --limit 50          # omit --limit for all frames
```

Writes under `local-data/.../output/comparison/groundingdino/{combined,players,logos}/`.

### YOLO

```bash
# Use the env with ultralytics; device mps|cuda|cpu
python yolo/yolo_labeler_player.py \
  --directory athletico_real_2023/close_ups \
  --device mps \
  --conf 0.25

python yolo/yolo_labeler_logo.py \
  --directory athletico_real_2023/close_ups \
  --device mps \
  --conf 0.25
```

Writes under `local-data/.../output/comparison/yolo/{players,logos}/`.

### Orchestrator

```bash
python run_bakeoff.py --directory athletico_real_2023/close_ups --limit 50 --dino-model low
python run_bakeoff.py --skip-train --limit 50          # weights already present
python run_bakeoff.py --skip-train --skip-dino         # YOLO only
```

---

## 7. How to compare results

Open the **same filename** in both trees, e.g. `frame_00196s_00.jpg`:

```text
.../comparison/groundingdino/players/annotated_frames/frame_00196s_00.jpg
.../comparison/yolo/players/annotated_frames/frame_00196s_00.jpg

.../comparison/groundingdino/logos/annotated_frames/frame_00196s_00.jpg
.../comparison/yolo/logos/annotated_frames/frame_00196s_00.jpg
```

Matching `txt_labels/*.txt` hold YOLO-format boxes for metrics / QA.

**Rough timings observed (Apple Silicon, 341 close-ups):** YOLO logo+player ≈ tens of seconds on MPS; DINO joint on CPU ≈ 15–25 minutes.

---

## 8. Script index

| Script | Role |
|--------|------|
| `paths.py` | Resolves `local-data` roots; comparison output helpers |
| `label_io.py` | Write YOLO TXT + draw boxes |
| `frame_extractor.py` | Video → wide/close frames |
| `auto_labeler.py` | Joint DINO → comparison/groundingdino/ |
| `auto_labeler_logo_generalized.py` | Shim / logos-only helper (legacy path) |
| `run_bakeoff.py` | Optional train + DINO + YOLO labelers |
| `yolo/config.py` | Roboflow IDs, epochs, weight paths |
| `yolo/download_datasets.py` | Pull player + logo Roboflow sets |
| `yolo/build_merged_logo_dataset.py` | Remap all logo classes → `0` / `logo` |
| `yolo/train_player_yolo.py` / `train_logo_yolo.py` | Ultralytics train → `weights/yolo/*/best.pt` |
| `yolo/yolo_labeler_player.py` / `yolo_labeler_logo.py` | Inference → comparison/yolo/ |
| `YOLO_Bakeoff_Train_Colab.ipynb` | GPU train + zip weights |

---

## 9. Adding jersey-context logos later (e.g. Riyadh Air)

Logo YOLO is **class-agnostic**. To improve detection of missing chest sponsors:

1. Collect **football kit / close-up** frames (Atlético kits for Riyadh Air), not only clean brand PNGs.
2. Label YOLO boxes: `0 cx cy w h` (normalized).
3. Layout like Roboflow: `extra_logos/train|valid/{images,labels}/` under `data/yolo_train/`.
4. Extend `build_merged_logo_dataset.py` (or `manifest.json`) to merge that folder, then retrain `train_logo_yolo.py` and replace `weights/yolo/logo_detector/best.pt`.

General “logo on a car / chrome PNG” data transfers poorly to logos on shirts; prefer kit footage + QA’d boxes.

---

## 10. Agent checklist (cold start)

1. Clone `csc490-group2`, checkout `omar-benchmark-groundingdino-yolo`.
2. Create `../local-data/groundingdino/data_collection/{weights,data/input/raw_videos}`.
3. Place DINO `.pth` files under `weights/`; place or train YOLO `best.pt` under `weights/yolo/...`.
4. Place match mp4 under `data/input/raw_videos/`; run `frame_extractor.py` → `close_ups/`.
5. Install envs (ultralytics + groundingdino) as above.
6. Run `auto_labeler.py` + both `yolo_labeler_*.py` (or `run_bakeoff.py --skip-train`).
7. Diff annotated frames under `data/output/comparison/`.

Do **not** commit `local-data/`, `*.pt`, `*.pth`, or Roboflow downloads. Commit script/doc changes only inside `csc490-group2/`.
