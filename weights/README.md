# Trained weights

Trained by the CSC490 Group 2 pipeline (`Untitled10_v3` notebook: Player Detector → Logo Detector → Doctor).

| File | Role | Arch | Size | SHA256 |
|---|---|---|---|---|
| `player.pt` | **Player detector** — localizes players / goalkeepers (gives the human OBB used to keep jersey logos and reject ad boards) | YOLOv8n, fine-tuned | 6.5 MB | `c22fa7cb…b480dc` |
| `logo.pt` | **Logo detector** — class-agnostic ("where does a logo look like"), trained on the merged Roboflow sponsor datasets + LogoDet-3K + synthetic jersey composites | YOLOv8, fine-tuned, 1 class | 22.7 MB | `c4978422…d9686` |
| `resnet50_doctor_v2.pth` | **Doctor** — brand classifier over the full merged logo vocabulary. Checkpoint dict contains `model_state`, `class_to_idx`, `idx_to_class`, `num_classes`, `temperature` (fitted on val), `other_key` | ResNet-50 | 134.7 MB | `db2145d6…2e0d6` |

> `resnet50_doctor_v2.pth` is stored with **Git LFS** (GitHub blocks single blobs > 100 MB).
> Clone with `git lfs install && git lfs pull`, otherwise you get a 130-byte pointer file.

## Usage

```python
import torch
from pathlib import Path
from ultralytics import YOLO

W = Path("weights")
player = YOLO(W / "player.pt")
logo = YOLO(W / "logo.pt")

ckpt = torch.load(W / "resnet50_doctor_v2.pth", map_location="cpu", weights_only=False)
idx_to_class = {int(k): v for k, v in ckpt["idx_to_class"].items()}
temperature = float(ckpt.get("temperature", 1.0))

import torch.nn as nn
from torchvision.models import resnet50
doctor = resnet50()
doctor.fc = nn.Linear(doctor.fc.in_features, len(idx_to_class))
doctor.load_state_dict(ckpt["model_state"])
doctor.eval()
```

In `Sponsor_Inference_Pipeline.ipynb` just point the config cell at these:

```python
PLAYER_WEIGHTS = Path("weights/player.pt")
LOGO_WEIGHTS   = Path("weights/logo.pt")
DOCTOR_WEIGHTS = Path("weights/resnet50_doctor_v2.pth")
```

## Notes / known limits

- **Doctor is closed-set** (~4.9k brand classes). Brands outside the training vocabulary are forced to the nearest known class; use the confidence / top1-top2 margin thresholds (or an `other` rejection class) to fall back to `unknown`.
- The logo vocabulary inherited from LogoDet-3K contains noisy class names (abbreviations, misspellings). Filter the allowed brand list before reporting if you need clean output.
- `logo.pt` is class-agnostic (single class `logo`) — it does not predict brands.
