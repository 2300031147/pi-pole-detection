# Pi Pole Detection (Rule-Based, Zero-Training)

This project implements a **zero-training** pole voltage inference pipeline designed for Raspberry Pi-class devices. It uses:

* **Pretrained geometry** (edges + lines) rather than fine-tuning.
* **Feature extraction** from pole structure (crossarms, levels, spacing).
* **Rule-based inference** to classify voltage categories.

The pipeline is intentionally simple and explainable—ideal for projects with only a handful of images.

## Features

* Edge detection + Hough line transforms for pole geometry.
* Horizontal crossarm clustering to estimate vertical levels.
* Basic insulator detection via contour roundness.
* Rule engine with confidence scoring.
* CLI tool for batch image processing.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run on a single image
pi-pole-detect --image path/to/image.jpg --output outputs/result.json

# Run on a folder of images
pi-pole-detect --image-dir path/to/images --output-dir outputs
```

## Project Layout

```
.
├── configs/
│   └── default.yaml
├── src/
│   └── pi_pole_detection/
│       ├── cli.py
│       ├── config.py
│       ├── detect.py
│       ├── features.py
│       └── rules.py
├── tests/
│   └── test_features.py
└── README.md
```

## Notes

* This project focuses on **explainability** over black-box ML.
* The default rules are meant to be edited based on local standards.
* Add more features incrementally as you collect additional images.
