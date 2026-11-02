# Detector Bake-off (Workflow 1)

Compare **GroundingDINO** vs **trained YOLOv8** player/logo localization on the same close-ups.

## Layout

```text
local-data/groundingdino/data_collection/
  data/input/framed_videos/athletico_real_2023/close_ups/
  data/output/comparison/
    groundingdino/{combined,players,logos}/{annotated_frames,txt_labels}/
    yolo/{players,logos}/{annotated_frames,txt_labels}/
  weights/
    groundingdino_*.pth
    yolo/{player_detector,logo_detector}/best.pt
```

## Setup

Run commands from `csc490-group2/groundingdino/data_collection/` (or use absolute paths).

```bash
pip install ultralytics roboflow pyyaml opencv-python torch torchvision
# GroundingDINO package must already be installed for auto_labeler*.py
```

## Train YOLO (once)

### Option A — Google Colab (recommended)

1. Open [`YOLO_Bakeoff_Train_Colab.ipynb`](YOLO_Bakeoff_Train_Colab.ipynb) in [Colab](https://colab.research.google.com/) (File → Upload notebook).
2. Runtime → GPU → Run all.
3. Download `yolo_bakeoff_weights.zip`.
4. Unzip into `local-data/groundingdino/data_collection/weights/yolo/` so you have:
   - `player_detector/best.pt`
   - `logo_detector/best.pt`

### Option B — Local

```bash
cd csc490-group2/groundingdino/data_collection/yolo
python download_datasets.py
python build_merged_logo_dataset.py
python train_player_yolo.py          # default 25 epochs
python train_logo_yolo.py            # default 5 epochs
```

Weights copy to `local-data/.../weights/yolo/*/best.pt`.

## Inference bake-off (e.g. 50 frames)

```bash
cd csc490-group2/groundingdino/data_collection

# GroundingDINO (one script: person/player/people/man + logo/text)
python auto_labeler.py --directory athletico_real_2023/close_ups --model low --limit 50

# YOLO (requires trained best.pt)
python yolo/yolo_labeler_player.py --directory athletico_real_2023/close_ups --limit 50
python yolo/yolo_labeler_logo.py --directory athletico_real_2023/close_ups --limit 50
```

Or one shot (train + both inferences):

```bash
python run_bakeoff.py --directory athletico_real_2023/close_ups --limit 50 --dino-model low
python run_bakeoff.py --skip-train --limit 50   # after weights exist
```

## Compare

Open the same `frame_XXXXX.jpg` under:

- `comparison/groundingdino/players/annotated_frames/`
- `comparison/yolo/players/annotated_frames/`

and the matching `logos/` pair.
