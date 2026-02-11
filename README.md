# PER - Object Detection on Thermal Images

This repository contains the PER project for object detection with YOLO on thermal images.

## Quick Start

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r models/yolov5/requirements.txt
pip install ultralytics roboflow
```

## Project Tree

```
PER/
├── README.md                    # This guide
├── data.yaml                    # YOLO dataset config
├── yolov8s.pt                   # Base YOLOv8 weights
├── annotations_trainval2017.zip # Optional COCO annotations archive
│
├── archive_v1/                  # Phase 1 work (legacy)
│   ├── scripts/
│   ├── notebooks/
│   └── data/
│
├── datasets/                    # Datasets
│   └── PER_ENT/                 # Main dataset
│       ├── images/              # train/ and val/
│       ├── labels/              # train/ and val/
│       └── annotations/
│
├── benchmarks/                  # Benchmark outputs
│   ├── metrics/
│   ├── results/
│   └── reports/
│
├── docs/                        # Documentation
│
├── experiments/                 # Experiments
│
├── models/                      # YOLO code and weights
│   └── yolov5/
│
├── src/                         # Source code
│   ├── inference/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
│
└── tests/                       # Tests
    ├── unit/
    ├── integration/
    └── models/
```

## Dataset Setup

Just run the training command below. It will automatically download the dataset from Roboflow.

## Training

```bash
# Roboflow training script
python src/training/training_model.py
```

## Evaluation

- YOLOv5 results are saved in `models/yolov5/runs/`.
- For benchmark reports, see `benchmarks/README.md`.

## Inference

Check `src/inference/` for Raspberry Pi and video inference scripts.

## Notes

- `src/training/training_model.py` downloads the dataset from Roboflow.
