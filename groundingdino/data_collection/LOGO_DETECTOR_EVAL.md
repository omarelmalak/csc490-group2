# Logo Detector Evaluation — 2023 Atlético vs Real Madrid

This note records the real-match check of the class-agnostic YOLO logo detector,
including the 5-epoch checkpoint and the corrected 25-epoch checkpoint.

## What BAKEOFF already measures

`BAKEOFF.md` compares two detectors on extracted match frames:

1. `yolo/yolo_labeler_player.py` — player/people boxes from the trained YOLO player detector.
2. `yolo/yolo_labeler_logo.py` — logo boxes from the trained YOLO logo detector.
3. `auto_labeler.py` — GroundingDINO person + logo/text proposals.

The output is qualitative: matching annotated JPGs and YOLO TXT labels under
`local-data/groundingdino/data_collection/data/output/comparison/`.
There is **no ground truth for match frames**, so match-footage recall/precision
cannot be computed directly from `BAKEOFF.md` alone.

## How to reproduce the real-match check

Benchmark video: <https://www.youtube.com/watch?v=yxM140uJEZI>
(Atlético de Madrid 3–1 Real Madrid, 2023/24 full match).

```bash
# 1. Download a 10-minute 720p slice (video only is enough)
cd local-data/groundingdino/data_collection/data/input/raw_videos
yt-dlp -f 136 \
  --download-sections "*00:15:00-00:25:00" \
  --force-keyframes-at-cuts \
  -o athletico_real_2023.mp4 \
  "https://www.youtube.com/watch?v=yxM140uJEZI"

# 2. Extract 1 frame / 2 s; frame_extractor separates wide shots from close-ups
cd ../../..   # local-data/groundingdino/data_collection
python ../../../groundingdino/data_collection/frame_extractor.py \
  --video athletico_real_2023.mp4 --minutes 10 --interval 2
# -> data/input/framed_videos/athletico_real_2023/{close_ups,wide_shots}

# 3. Put the trained logo weights in the expected place
cp ~/Downloads/best.pt \
  weights/yolo/logo_detector/best.pt

# 4. Run the bake-off logo labeler
cd ../../../groundingdino/data_collection
python yolo/yolo_labeler_logo.py \
  --directory athletico_real_2023/close_ups \
  --conf 0.25 --limit 62
# -> data/output/comparison/yolo/logos/{annotated_frames,txt_labels}
```

## Current checkpoint

The corrected `Downloads/best.pt` is a **25-epoch**, class-agnostic,
single-class logo detector trained on the same merged dataset:

```text
task=detect, nc=1, names={0: 'logo'}
train_args.data = /content/data/merged_logo_detector/data.yaml
train_args.epochs = 25
train_metrics: precision=0.867, recall=0.754, mAP50=0.8246, mAP50-95=0.4481
mAP50 over epochs 1..25 = 0.621, 0.570, 0.625, 0.684, 0.722, ...
                          0.802, 0.811, 0.824, 0.807, 0.8246
```

The earlier 5-epoch checkpoint is kept for comparison:

```text
train_args.epochs = 5
train_metrics: precision=0.820, recall=0.693, mAP50=0.790, mAP50-95=0.377
```

## Quantitative diagnostics

Script: `yolo/eval_logo_detector.py` (class-agnostic matching, IoU=0.5).

```bash
cd groundingdino/data_collection/yolo

# Sponsor-detector-2 valid split: logos on players / jerseys
python eval_logo_detector.py \
  --weights ../../../local-data/groundingdino/data_collection/weights/yolo/logo_detector/best_25ep_merged_1class.pt \
  --images Sponsor-detector-2/valid/images \
  --labels Sponsor-detector-2/valid/labels \
  --limit 125 --conf 0.001

# Generic logo detection sample (Logo-Detection-4 only has a train split)
python eval_logo_detector.py \
  --weights ../../../local-data/groundingdino/data_collection/weights/yolo/logo_detector/best_25ep_merged_1class.pt \
  --images Logo-Detection-4/train/images \
  --labels Logo-Detection-4/train/labels \
  --limit 200 --conf 0.001
```

### Sponsor-detector-2 valid (logos on players)

| Epochs | conf | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| 5 | 0.05 | 0.887 | 0.295 | 0.442 |
| 5 | 0.25 | 0.749 | 0.743 | 0.746 |
| 5 | 0.50 | 0.495 | 0.915 | 0.642 |
| **25** | **0.05** | **0.874** | **0.367** | **0.517** |
| **25** | **0.10** | **0.849** | **0.509** | **0.637** |
| **25** | **0.25** | **0.813** | **0.764** | **0.788** |
| **25** | **0.50** | **0.592** | **0.917** | **0.720** |

### Logo-Detection-4 sample (generic logos)

Note: this dataset has only a train split, so these are 200 sampled training
images. Even on training data, the 5-epoch model was weak.

| Epochs | conf | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| 5 | 0.05 | 0.535 | 0.161 | 0.248 |
| 5 | 0.25 | 0.182 | 0.540 | 0.272 |
| 5 | 0.50 | 0.032 | 0.667 | 0.061 |
| **25** | **0.05** | **0.754** | **0.324** | **0.453** |
| **25** | **0.10** | **0.668** | **0.463** | **0.547** |
| **25** | **0.25** | **0.540** | **0.706** | **0.612** |
| **25** | **0.50** | **0.332** | **0.886** | **0.482** |

Confidence of matched boxes (IoU >= 0.5):

| Checkpoint | Dataset | median | p90 |
|---|---|---:|---:|
| 5 epoch | Sponsor valid | 0.046 | 0.601 |
| 5 epoch | Generic sample | 0.015 | 0.210 |
| **25 epoch** | **Sponsor valid** | **0.072** | **0.691** |
| **25 epoch** | **Generic sample** | **0.034** | **0.593** |

The 25-epoch run fixes a large part of the generic-logo underfitting: generic
recall at conf=0.25 goes from 0.182 to 0.540, and precision from 0.540 to 0.706.

## Match-footage check

Same 62 close-up frames and 238 wide-shot frames from the benchmark video:

| Checkpoint | Footage | conf>=0.05 | conf>=0.10 | conf>=0.25 | conf>=0.50 | median per-frame max conf | p90 |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 epoch | close-ups | 285 | 121 | 40 | 9 | 0.199 | 0.520 |
| **25 epoch** | **close-ups** | **242** | **107** | **47** | **13** | **0.203** | **0.580** |
| 5 epoch | wide shots | 210 | 85 | 33 | 10 | 0.035 | 0.177 |
| **25 epoch** | **wide shots** | **458** | **208** | **58** | **14** | **0.095** | **0.358** |

At the default `conf=0.25` the 25-epoch model goes from 40 → 47 boxes on
close-ups and from 33 → 58 on wide shots. Wide-shot confidence roughly triples.

## Isolated logo crop test

A tight crop around one logo makes the object occupy a large fraction of the
input and removes the player/field context. Tested on 120 random crops per
source (25% margin around the first GT box):

| Checkpoint | Crop source | median best IoU | median conf | conf>=0.25 & IoU>=0.5 |
|---|---|---:|---:|---:|
| 5 epoch | Sponsor valid | 0.682 | 0.0038 | 1 / 120 |
| 5 epoch | Generic sample | 0.733 | 0.0062 | 0 / 120 |
| **25 epoch** | **Sponsor valid** | **0.704** | **0.0099** | **20 / 120** |
| **25 epoch** | **Generic sample** | **0.763** | **0.0272** | **26 / 120** |

The 25-epoch model is still less confident on isolated crops than on full match
frames, but this failure mode is dramatically reduced. It is now roughly
scale/context-robust enough to detect a meaningful fraction of single-logo
images, unlike the 5-epoch model.

## Why the original 5-epoch model looked person-biased

Training label statistics from the two source datasets:

| Dataset | Train images | Train boxes | Classes | Median box area fraction |
|---|---:|---:|---:|---:|
| Sponsor-detector-2 | 2488 | 7367 | 4 | 0.0024 |
| Logo-Detection-4 | 1374 | 1050 | 137 | 0.0124 |

About **87.5%** of training boxes come from the four sponsor logos on player
photos. With only 5 epochs, the model learned the dominant mode first:
small sponsor logos on jerseys / broadcast close-ups. Training to 25 epochs lets
it fit the generic-logo tail much better.

## Hyundai is not in the detector training labels in this branch

A direct search of the two detector datasets used by this branch finds **no
`hyundai` class and no Hyundai label** in either `Sponsor-detector-2` or
`Logo-Detection-4`. `Sponsor-detector-2` only has `Adidas`, `Emirates`, `Nike`,
`Spotify`; `Logo-Detection-4` has 137 generic logo classes but not Hyundai.

If the Hyundai images are only in the **classification** dataset (LogoDet-3K /
Logo-2341 / custom Drive folder), they never train the logo detector. The
class-agnostic detector can only generalize from the generic logo classes it
actually saw.

Class counts in the detector source sets:

```text
Sponsor-detector-2: Adidas 1414, Emirates 1154, Nike 2476, Spotify 2871 boxes
Logo-Detection-4:   137 classes, 1050 boxes total
```

## Other checkpoints found in git history

These are useful for comparison but are **not** the same experiment:

| Checkpoint | Classes | Train data | Epochs | close-ups boxes @conf .25 | wide-shot boxes @conf .25 |
|---|---:|---|---:|---:|---:|
| `best_5ep_merged_1class.pt` | 1 (`logo`) | merged Sponsor + Logo-Detection-4 | 5 | 40 | 33 |
| **`best_25ep_merged_1class.pt` / current `best.pt`** | **1 (`logo`)** | **merged Sponsor + Logo-Detection-4** | **25** | **47** | **58** |
| `best_25ep_375class.pt` (git history) | 375 | LogoDet + Sponsor | 25 | 44 | 58 |
| `best_20ep_4class_sponsor.pt` (git history) | 4 sponsor | Sponsor only | 20 | 53 | 56 |

The 25-epoch merged 1-class checkpoint is the relevant one for the current
class-agnostic pipeline. The 375-class checkpoint is more confident on match
frames but is not class-agnostic and trained on different generic classes; the
4-class checkpoint only knows the four sponsor brands.

## Practical next steps

1. **Keep the 25-epoch merged checkpoint as the baseline.** It improves both
   generic-logo recall and match-footage confidence.
2. **Still rebalance the merged detector set.** Sponsor/player boxes are ~7×
   more numerous than generic-logo boxes. Oversampling `Logo-Detection-4` or
   two-stage training (generic pretrain → match fine-tune) is the next win.
3. **Add match-domain hard positives/negatives.** Label advertising boards,
   overlays, and non-player logos from the benchmark video, or use
   GroundingDINO to pre-label and correct.
4. **Consider tiled inference for wide shots.** At 640 the wide-shot confidence
   remains much lower than close-ups. `imgsz=1280` on the generic set increased
   recall at low confidence but also increased false positives, so tune on a
   fixed validation set rather than assuming higher resolution is always better.
5. **Keep the detector class-agnostic.** Brand identity belongs to the Doctor.
