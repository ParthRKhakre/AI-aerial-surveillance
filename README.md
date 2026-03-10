# Dual-Window AI Aerial Surveillance Real-Time Identifier System

## 1) Project overview and goals
This project implements a production-style, modular Python pipeline for aerial/satellite/drone video analytics.

Core goals:
- Real-time object detection with YOLOv8/YOLOv9-compatible workflow.
- Persistent multi-object tracking with ByteTrack (DeepSORT can be swapped in with the same interface).
- Dual-window UI: **RAW FEED** on the left and **AI ANALYSIS** on the right.
- Rich overlays: class-colored boxes, class + confidence labels, tracking IDs, trajectory trails, and FPS.
- Optional FastAPI endpoint for frame-by-frame detection inference.

## 2) System architecture (ASCII)
```text
Video/RTSP Input
      |
      v
[Frame Extraction + Preprocessing] --(optional frame skipping/batching)--> [YOLO Detection]
      |                                                                |
      |                                                                v
      +--------------------------------------------------------> [Object Class Labels]
                                                                       |
                                                                       v
                                                               [ByteTrack / DeepSORT]
                                                                       |
                                                                       v
                                                         [Visualization + Overlay Engine]
                                                                       |
                                                                       v
                                                      [Dual-Window UI (OpenCV / PyQt)]
```

## 3) Dataset preparation guide
### Recommended aerial datasets
- **DOTA**: https://captain-whu.github.io/DOTA/dataset.html
- **xView**: https://xviewdataset.org/
- **VisDrone**: https://github.com/VisDrone/VisDrone-Dataset
- **UAVDT**: https://sites.google.com/view/daweidu/projects/uavdt

### Folder staging
```text
data/
  raw/
    dota/
    xview/
    visdrone/
    uavdt/
  processed/
    images/{train,val,test}
    labels/{train,val,test}
```

### Annotation conversion to YOLO format
YOLO label per line:
```text
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

Use provided converter for COCO-style exports:
```bash
python scripts/convert_coco_to_yolo.py \
  --coco-json data/raw/xview/annotations/train.json \
  --images-dir data/raw/xview/images \
  --labels-out data/processed/labels/train
```

### Train/val/test split
- Recommended split: `70/20/10` or `80/10/10` depending on dataset size.
- Keep scene diversity balanced across splits (urban, rural, ports, highways).
- Avoid geographic leakage: do not put near-identical neighboring tiles in both train and test.

### Aerial-specific augmentation recommendations
- **Small-object support**: larger input resolution (`imgsz=960` or `1280`).
- **Top-down viewpoint robustness**: slight rotations + perspective transforms.
- **Dense-scene robustness**: mosaic + mixup with controlled intensity.
- **Lighting/weather variation**: HSV augmentation, synthetic haze, contrast jitter.

## 4) Model selection and training guide
### Detector
- Default: **YOLOv8** via Ultralytics (also compatible workflow for YOLOv9 variants if exported similarly).
- Start from COCO-pretrained weights (`yolov8m.pt` for stronger baseline).

### Tracker
- Default integrated tracker: **ByteTrack** (`src/tracking/byte_tracker_wrapper.py`).
- DeepSORT can be swapped in by replacing tracker module with same `update()` contract.

### Train command
```bash
python scripts/train_yolo.py --config config/train_aerial_yolov8.yaml
```

### Fine-tuning workflow
1. Initialize from pretrained COCO checkpoint.
2. Train on merged aerial classes or per-dataset then joint-finetune.
3. Validate on held-out split using `scripts/evaluate.py`.
4. Export best checkpoint to deployment formats (`ONNX`, `TensorRT engine`).

## 5) Step-by-step implementation guide
1. Create environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Install CUDA PyTorch build (example):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. Place weights at `models/yolov8n.pt` (or custom `best.pt`).
4. Run real-time system:
   ```bash
   python -m src.main --source data/sample.mp4 --weights models/yolov8n.pt --imgsz 960
   ```
5. Keyboard shortcuts for overlays are documented in `ui/keyboard_controls.md`.

## 6) Full working Python code modules
- Video loader: `src/preprocessing/video_loader.py`
- Detector: `src/detection/yolo_detector.py`
- Tracker (ByteTrack): `src/tracking/byte_tracker_wrapper.py`
- Visualization engine: `src/visualization/renderer.py`
- Main dual-window application: `src/main.py`
- Optional API backend: `api/fastapi_app.py`

Run API:
```bash
uvicorn api.fastapi_app:app --host 0.0.0.0 --port 8000
```

## 7) UI implementation details
- Left panel is always raw frame with `RAW FEED` title.
- Right panel is annotated frame with `AI ANALYSIS` title.
- Overlays include:
  - class-colored bounding boxes,
  - labels with class name + confidence,
  - persistent tracking ID,
  - trajectory trails (last N centers),
  - FPS counter.
- Toggle overlays in real time:
  - `b` boxes
  - `l` labels/confidence
  - `i` IDs
  - `t` trails
  - `f` FPS
  - `q` quit

## 8) Optimization techniques
### GPU acceleration
- Auto device selection uses CUDA when available.
- FP16 inference enabled on CUDA (`half=True`) to reduce latency.

### TensorRT export
From Python:
```python
from src.detection.yolo_detector import YOLODetector
YOLODetector("models/best.pt").export_tensorrt(dynamic=True, workspace=4)
```

### Frame skipping
- Use `--skip N` to process every `N+1` frame for weak hardware.
- Preserves real-time display under limited GPU/CPU throughput.

### Quantization
- FP16 is integrated for inference.
- INT8 deployment is recommended during TensorRT calibration for edge devices.

### Batch inference
- For offline processing, adapt detector to pass frame lists into Ultralytics `predict` for batched throughput.

## 9) Testing and evaluation
### Example aerial sources
- VisDrone benchmark sequences (download from official site).
- UAVDT videos.
- Public YouTube drone footage converted to local MP4 for reproducible testing.

### Evaluation scripts
```bash
python scripts/evaluate.py --model models/best.pt --data config/aerial_data.yaml --video data/sample.mp4
python scripts/evaluate.py --model models/best.pt --data config/aerial_data.yaml --gt data/gt.csv --pred data/pred.csv
```
Outputs:
- Detection `mAP50` and `mAP50-95`
- Tracking `MOTA`, `MOTP`, and ID switches
- Average inference FPS

## 10) Future enhancement roadmap
- Threat-detection alert engine with per-class trigger rules.
- Automatic target-priority scoring (speed, size, geofence relevance).
- GPS/map overlay and geo-referenced event logging.
- Multi-camera tiled command-center view.
- Direct drone SDK integrations (DJI/ArduPilot/PX4).
- Cloud-native relay: RTSP ingest + WebRTC egress + scalable analytics workers.
