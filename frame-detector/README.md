# Soccer frame detector

## Goal

This component selects broadcast frames that are useful for downstream soccer
jersey-sponsor logo detection.

The desired output is a binary decision:

- **`close_up` / yes:** the play is shown closely enough that a viewer could
  plausibly make out player facial features or clearly see logos on jerseys.
- **`not_close_up` / no:** everything else, including bird's-eye or wide shots,
  audience and dugout shots, replays dominated by graphics, stadium views, and
  frames with no relevant player.

A frame should be rejected if players are too small, the image is too blurred or
occluded for logo recognition, or the image is dominated by irrelevant content.
The model is not intended to identify whether a frame is aesthetically
interesting; it is intended to protect the downstream logo detector from
unusable inputs.

## Evaluating the initial heuristics

The labeled folders can be evaluated with:

```powershell
python -m pip install opencv-python ultralytics
python frame-detector\evaluate_heuristics.py
```

The evaluator uses the default labeled match directory under `data/`, excludes
`uncertain` frames from the metrics, and writes per-frame features and
predictions to:

```text
data/frame-detector/heuristic_predictions.csv
```

The initial rules are:

- reject green ratio `< 5%` or `> 85%`
- otherwise accept if the largest confident person box is at least `25%` of
  the frame height
- use YOLO's COCO `person` class (`class 0`) with confidence at least `0.50`

All thresholds are configurable. For example:

```powershell
python frame-detector\evaluate_heuristics.py `
  --green-low 0.05 `
  --green-high 0.85 `
  --person-height 0.25 `
  --person-confidence 0.50
```

The output CSV includes the label, prediction, green ratio, number of detected
people, largest relative person height, and whether the green filter rejected
the frame. This makes false positives and false negatives inspectable instead
of reducing the experiment to one accuracy number.

To tune thresholds without running YOLO again, use the cached CSV:

```powershell
python frame-detector\tune_heuristics.py
```

This searches a small grid of green lower/upper bounds and person-height
thresholds. It assigns consecutive frame-ID blocks to validation and test
sets, so adjacent frames are not randomly mixed between them. The best
thresholds are selected using validation F1, then reported once on the held-out
test blocks. Use `--beta 2` if preserving usable `close_up` frames is more
important than precision; do not select thresholds using the test result.

Because the current labels come from one match, this is still a preliminary
tuning result. The final threshold should be checked on at least one separate
match or broadcast style.

### Heuristic tuner results

The tuner was run on the currently labeled frames, excluding `uncertain`
examples. The selected thresholds were:

```text
green_low=0.02
green_high=0.90
person_height=0.20
```

| Evaluation set | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tuned validation | 0.841 | 1.000 | 0.356 | 0.409 | 36 | 0 | 65 | 308 |
| Tuned test | 0.931 | 1.000 | 0.550 | 0.604 | 33 | 0 | 27 | 333 |
| Default validation | 0.822 | 1.000 | 0.277 | 0.324 | 28 | 0 | 73 | 308 |
| Default test | 0.913 | 1.000 | 0.433 | 0.489 | 26 | 0 | 34 | 333 |

The tuned thresholds improve test accuracy, recall, and F1 over the original
`green_low=0.05`, `green_high=0.85`, and `person_height=0.25` settings. The
most meaningful improvement is recall: the tuned rule retains 55.0% of labeled
`close_up` frames compared with 43.3% for the default rule. Both configurations
have perfect precision on this test split, but both still miss many usable
frames. This means the heuristic is conservative and useful as an interpretable
baseline, but it should not yet be treated as a complete close-up classifier.

These results are preliminary because the data comes from one match and the
labels are temporally related. The tuned thresholds must be evaluated on
additional matches before being adopted for the logo-detection pipeline.

## Gold-standard labels

Label individual frames using the criteria above. During annotation, an
`uncertain` or `ambiguous` status may be used for borderline examples rather
than forcing an unreliable yes/no decision. Ambiguous frames should be reviewed
by a second annotator or excluded from the initial evaluation set.

Use a CSV with at least:

```text
frame_path,label,video_id,timestamp
```

Labels must be assigned from the image itself, not from detector predictions or
the proposed heuristics. For a first proof of concept, target approximately
500 frames with a reasonably balanced number of `close_up` and `not_close_up`
examples. Include different matches, camera styles, lighting conditions, and
shot types.

Split by `video_id` or match, never by random individual frames. Neighboring
frames are highly correlated; putting frames from one sequence in both
training and validation would produce misleading results.

## Proof-of-concept pipeline

The first implementation should be an interpretable heuristic baseline:

```text
RGB frame
  -> HSV pitch-green mask
  -> reject obvious audience/non-field frames
  -> YOLO person detection
  -> measure detected person sizes
  -> accept or reject frame
  -> pass accepted frames to the logo detector
```

### 1. Green pre-filter

Convert the frame to HSV and mask pixels matching the pitch green. Start with
these thresholds, then tune them on a validation set:

- green ratio `< 5%`: likely audience, dugout, graphics, or another
  non-field shot; reject
- green ratio `> 85%`: likely a high bird's-eye field view; reject

These are rejection heuristics, not ground truth. Lighting, shadows, artificial
turf, uniforms, and camera framing can make green percentage unreliable.

### 2. Person-size filter

Run an off-the-shelf COCO-pretrained YOLO detector on frames that survive the
green filter. Keep only the `person` detections above a confidence threshold.
For each detection, calculate:

```text
relative_height = person_box_height / frame_height
relative_area = person_box_area / frame_area
```

Initially classify a frame as `close_up` when at least one sufficiently
confident person has a relative height around `20%–30%` or greater. Reject
frames where all detected people are tiny. A minimum relative area and detector
confidence should also be considered.

The person-height threshold is camera-dependent and must be measured against
the labeled data. A large box does not guarantee a visible logo: the player
may face away, be occluded, or be blurred.

### 3. Evaluation

Apply the initial rules unchanged to the labeled validation frames. Report:

- precision
- recall
- F1 score
- confusion matrix
- false negatives, especially usable frames incorrectly discarded

Use the same held-out match-level test set for the final comparison. Prefer
recall for usable logo frames when an extra downstream inference is cheaper
than losing a valid sponsorship exposure.

## Future improvements

The heuristic baseline can be compared with a learned binary image classifier.
The classifier can learn cues that simple thresholds miss, such as camera angle,
blur, audience composition, player orientation, and broadcast graphics.

A stronger final system may combine:

```text
green pre-filter
  -> YOLO person detections
  -> heuristic features (green ratio, box height, box area, confidence)
  -> binary classifier or calibrated decision rule
  -> player/torso crop
  -> jersey-logo detector
```

Temporal smoothing can reduce flickering decisions: nearby frames from the same
shot should generally receive consistent classifications. Accepted player
detections should be cropped to the relevant torso or player region before
being passed to the logo detector.

## Data sources and `SN-Features`

The Hugging Face dataset
[SoccerNet/SN-Features](https://huggingface.co/datasets/SoccerNet/SN-Features)
does **not** contain RGB image frames. Its branches contain:

- ResNet or Baidu feature vectors
- Mask R-CNN player bounding-box metadata
- field-camera calibration metadata

The `player-boundingbox-maskrcnn` branch may help analyze player-size features,
but it cannot calculate the HSV pitch-green ratio, inspect facial expressions,
or run YOLO on pixels. The metadata is sampled at approximately 2 FPS and does
not include the corresponding video frames.

Run the local inspection script from the repository root:

```powershell
python frame-detector\inspect_soccernet_features.py
```

To download one game's player-box metadata:

```powershell
python -m pip install huggingface_hub

python frame-detector\inspect_soccernet_features.py `
  --download-game "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
```

For the actual proof of concept, obtain SoccerNet videos through their
authorized download workflow or use legally obtained broadcast footage, then
extract frames locally. Video access is separate from `SN-Features`. Keep
videos, extracted frames, and other datasets outside Git.
