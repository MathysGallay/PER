# Methods Used in the PER Project

This document lists the main methods used in the project, including the legacy phase in archive_v1 and the current pipeline.

## Phase 1 (archive_v1)

### Mini-COCO extraction (animals only)
- Source: COCO 2017 annotations.
- Method: filter annotations to animal classes and download only the required images.
- Output: a reduced dataset with YOLO labels for 10 animal classes.
- Script: `archive_v1/scripts/download_mini_coco_animals.py`.

### RGB to thermal-like conversion
- Downscale to Topdon TC001 resolution (256x192).
- Convert to grayscale and reduce texture (Gaussian blur + bilateral filter).
- Apply global contrast/brightness variation.
- Add thermal noise (Gaussian noise).
- Invert grayscale so objects appear hot.
- Apply JET colormap to simulate thermal colors.
- Script: `archive_v1/scripts/image_thermal_converter.py`.

### Thermal data augmentation
- Random horizontal and vertical flips.
- Random crop + resize back to original size.
- Random Gaussian blur.
- Add random noise.
- Script: `archive_v1/scripts/thermal_augmentation.py`.

### YOLOv5 training pipelines (notebooks)
- Early training tests and end-to-end pipeline experiments.
- Notebooks: `archive_v1/notebooks/pipeline_yolov5.ipynb` and `archive_v1/notebooks/pipeline_yolov5_restructured.ipynb`.

## Phase 2 (current pipeline)

### Dataset acquisition via Roboflow
- Training script connects to Roboflow, selects the project and version, and downloads the dataset in YOLOv8 format.
- Dataset is placed under `datasets/` and then used by the training call.
- Script: `src/training/training_model.py`.

### YOLO training (Ultralytics)
- Uses Ultralytics API to train with thermal resolution.
- Config uses `imgsz=(256, 192)`, batch size, patience, and a project output folder.
- Script: `src/training/training_model.py`.

### Motion heatmap generation
- Background subtraction (MOG2) on thermal stream.
- Morphological cleanup (erode/dilate), contour filtering by area.
- Accumulator for motion intensity over time and JET colormap overlay.
- Flask stream for live visualization.
- Script: `src/heatmap.py`.

### Hailo inference on Raspberry Pi (live stream)
- GStreamer pipeline with Hailo accelerator and YOLO post-processing.
- Flask server streams live inference frames.
- Script: `src/inference/detection_thermal_raspberry.py`.

### Hailo inference on video files (batch)
- GStreamer pipeline for video files, Hailo inference, and TCP streaming.
- Script: `src/inference/detection_video_raspberry.py`.

### On-demand inference with motion gating
- Split stream: heatmap branch for motion detection and inference branch gated by a valve.
- If motion is detected, inference is enabled; otherwise, it pauses after cooldown.
- Detections are logged to CSV and a dashboard provides two live feeds.
- Script: `src/inference/detection_thermal_main.py`.
