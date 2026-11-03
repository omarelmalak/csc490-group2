# Colab: full-pipeline evaluation on the logo-detector dataset

Runs:

```text
Sponsor-detector-2 image
  -> Player Detector
  -> Logo Detector
  -> containment (logo OBB inside player OBB)
  -> Doctor (ResNet-50)
  -> compare with GT brand
```

## What it reports

| Metric | Meaning |
|---|---|
| `detection_recall` | matched logo predictions / all GT logo boxes |
| `containment_pass_rate` | predictions that pass player containment / all GT boxes |
| `doctor_oracle_acc` | GT box crop -> Doctor top-1 accuracy (Doctor alone) |
| `doctor_pipeline_acc` | detector crop -> Doctor top-1 accuracy (conditional on match+containment) |
| `end_to_end_acc` | correct brand / all GT boxes (includes misses and filter failures) |

It also prints per-brand accuracy and a confusion matrix, and saves bad cases.

## Cell 1 — setup

```python
!pip -q install ultralytics roboflow pyyaml opencv-python

from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2 — download the evaluation script

```python
!wget -q -O eval_full_pipeline_on_detector_dataset.py \
  https://raw.githubusercontent.com/omarelmalak/csc490-group2/omar-benchmark-groundingdino-yolo/groundingdino/data_collection/eval_full_pipeline_on_detector_dataset.py
```

## Cell 3 — download Sponsor-detector-2 if you do not already have it

The script can auto-download it if `--data-root` does not contain `data.yaml`,
but you can also do it explicitly:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="A1UDb8W87l9RzmQ4CMlU")
ds = rf.workspace("syed-fathaullah").project("sponsor-detector").version(2).download("yolov8")
print(ds.location)   # use this as --data-root
```

## Cell 4 — run evaluation

Replace the Drive paths with your actual weight locations.

```python
!python eval_full_pipeline_on_detector_dataset.py \
  --data-root /content/Sponsor-detector-2 \
  --player-weights /content/drive/MyDrive/csc490/weights/yolo/player_detector/best.pt \
  --logo-weights /content/drive/MyDrive/csc490/weights/yolo/logo_detector/best.pt \
  --doctor-weights /content/drive/MyDrive/csc490/weights/resnet50_jersey_brand_doctor.pth \
  --splits test valid \
  --det-conf 0.25 \
  --match-iou 0.5 \
  --containment-threshold 0.95 \
  --out-dir /content/full_pipeline_eval
```

If the doctor checkpoint is a plain `state_dict` without `idx_to_class`, add:

```bash
--doctor-labels /content/drive/MyDrive/.../doctor_label_map.json
```

The label map JSON must contain `{"idx_to_class": {"0": "Adidas", ...}}`.

## Output

```text
/content/full_pipeline_eval/
  summary_test.json
  summary_valid.json
  records_test.jsonl
  records_valid.jsonl
  bad_cases_test/
  bad_cases_valid/
```

## Important: data leakage

If Doctor was trained on crops extracted from `Sponsor-detector-2/valid`,
then `valid` Doctor accuracy is inflated. Prefer `test`, or use a held-out
match-frame set that was never used for Doctor training.

## Notes

- If `--player-weights` is missing, the script falls back to `yolov8n.pt` COCO
  person class as a player detector.
- If `--logo-weights` is missing, it downloads the committed 25-epoch merged
  logo detector from `bakeoff_artifacts/`.
- The script maps GT brand names to Doctor class names with case/punctuation
  normalization. Brands missing from the Doctor vocabulary are reported under
  `unmapped_gt` and excluded from Doctor accuracy.
