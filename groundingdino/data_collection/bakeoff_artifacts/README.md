# Bake-off artifacts — logo detector (25 epoch)

Share these files with the team for the detector discussion.

## Files

| File | Description |
|---|---|
| `logo_detector_best_25ep_merged.pt` | YOLOv8n, class-agnostic logo detector, 25 epochs, trained on merged `Sponsor-detector-2 + Logo-Detection-4` |
| `match_frames_montage_25ep.jpg` | Top-6 highest-confidence close-up frames from the 2023 Atlético–Real Madrid benchmark, boxes drawn at `conf=0.05` |
| `eval_25ep_sponsor_valid.json` | Evaluation on `Sponsor-detector-2` valid split (125 images, 390 GT boxes) |
| `eval_25ep_generic_logo_sample.json` | Evaluation on 200 sampled `Logo-Detection-4` train images (187 GT boxes) |
| `eval_5ep_sponsor_valid.json` | Same sponsor evaluation, old 5-epoch checkpoint |
| `eval_5ep_generic_logo_sample.json` | Same generic evaluation, old 5-epoch checkpoint |

## Checkpoint metadata

```text
task=detect, nc=1, names={0: 'logo'}
train_args.data = /content/data/merged_logo_detector/data.yaml
train_args.epochs = 25

train_metrics:
  precision = 0.867
  recall    = 0.754
  mAP50     = 0.8246
  mAP50-95  = 0.4481
```

## Headline comparison (conf = 0.25, IoU = 0.5)

| Split | 5 epoch Recall | 25 epoch Recall | 5 epoch Precision | 25 epoch Precision |
|---|---:|---:|---:|---:|
| Sponsor-detector-2 valid | 0.749 | **0.813** | 0.743 | **0.764** |
| Logo-Detection-4 generic sample | 0.182 | **0.540** | 0.540 | **0.706** |

## Match footage (same 62 close-ups / 238 wide shots)

| Checkpoint | Footage | boxes @conf .25 | median per-frame max conf |
|---|---|---:|---:|
| 5 epoch | close-ups | 40 | 0.199 |
| 25 epoch | close-ups | **47** | 0.203 |
| 5 epoch | wide shots | 33 | 0.035 |
| 25 epoch | wide shots | **58** | **0.095** |

Full analysis: [`../LOGO_DETECTOR_EVAL.md`](../LOGO_DETECTOR_EVAL.md).

## Using the weights with the bake-off

```bash
cd csc490-group2/groundingdino/data_collection
mkdir -p ../../local-data/groundingdino/data_collection/weights/yolo/logo_detector
cp bakeoff_artifacts/logo_detector_best_25ep_merged.pt \
  ../../local-data/groundingdino/data_collection/weights/yolo/logo_detector/best.pt

python yolo/yolo_labeler_logo.py \
  --directory athletico_real_2023/close_ups \
  --conf 0.25 --limit 62
```

Or pass the weights directly:

```bash
python yolo/yolo_labeler_logo.py \
  --weights bakeoff_artifacts/logo_detector_best_25ep_merged.pt \
  --directory athletico_real_2023/close_ups \
  --conf 0.25 --limit 62
```
