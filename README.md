# Shirt Sponsor Logo Detector for ROI Analysis in Soccer Broadcasts
 
CSC490 Capstone, Group 2, University of Toronto Mississauga
 
## Overview
 
This project builds a deep learning system that detects sponsor logos on soccer jerseys in close-up broadcast footage and measures how visible each sponsor is. For every detected logo, the system reports the brand, where it appears on the player (chest, sleeve, back), and visibility metrics such as screen time and on-screen size. The goal is an objective, data-driven way to estimate the return on investment of shirt sponsorships.
 
## Approach
 
1. **Logo detection:** a YOLO-based object detector localizes logos in each frame.
2. **Brand classification:** each detected logo is assigned to a sponsor brand.


## Datasets
 
| Dataset | Source | Use |
|---|---|---|
| AFL Sponsor Logos | Roboflow Universe | Sponsor logos on jerseys in sports broadcasts |
| LogoDet-3K | [Kaggle](https://www.kaggle.com/datasets/lyly99/logodet3k) | Pretraining for general logo detection |
| Internally labeled frames | In progress | Fine-tuning, testing, and demo on soccer footage |
 
Datasets are not stored in this repository. Download them separately and place them under `data/` (ignored by git).

## Setup
 
```bash
git clone https://github.com/omarelmalak/csc490-group2.git
cd csc490-group2
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
## Team
 
- Omar El Malak
- Zhijie Jiang
- Haris Faisal
- Zakir Muhammad