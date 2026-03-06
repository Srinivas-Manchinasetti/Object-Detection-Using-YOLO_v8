# YOLOv8 Object Detection Project

This project provides a complete YOLOv8-based object detection system using the `ultralytics` Python package.  
You can:

- **Run detection** on images, videos, or webcam.
- **Real-time webcam detection** with live object tracking.
- **Train** a custom YOLOv8 model on your own dataset in YOLO format.

---

## 1. Environment setup

From your project folder:

```bash
cd yolov8_project

# (Optional) If you already created `yolo_env`, activate it:
# On PowerShell:
.\yolo_env\Scripts\Activate.ps1

# Or create a new virtualenv (example):
python -m venv yolo_env
.\yolo_env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> **Note**: If PyTorch fails to install via `pip install -r requirements.txt`,  
> install it first from the official instructions at: https://pytorch.org/get-started/locally/

---

## 2. Quick Start (Webcam Detection)

For instant real-time object detection:

```bash
cd yolov8_project
.\yolo_env\Scripts\Activate.ps1

# Run webcam detection
python webcam_detect.py
```

**Controls:**
- Press `q` in the detection window to stop
- Adjust confidence threshold: `python webcam_detect.py --conf 0.5`

---

## 3. Run object detection (inference)

The `detect.py` script lets you run YOLOv8 on:

- **Image file** (e.g. `images/example.jpg`)
- **Video file** (e.g. `videos/sample.mp4`)
- **Webcam** (`0` for default camera)

### Examples

```bash
cd yolov8_project
.\yolo_env\Scripts\Activate.ps1   # if not already active

# 1) Run on a single image
python detect.py --source path/to/image.jpg --show

# 2) Run on a video
python detect.py --source path/to/video.mp4 --show

# 3) Run on webcam (device 0)
python detect.py --source 0 --show
```

Main options:

- `--weights`: YOLOv8 weights file to use (default: `yolov8n.pt`).
- `--source`: Image/video path or webcam index (e.g. `0`).
- `--conf`: Confidence threshold (default: `0.25`).
- `--show`: Display results in a window (add this flag to see output).
- `--save`: Save result images/videos with predictions (default: enabled).

Output is saved under `runs/detect/...` by default (Ultralytics default).

---

## 4. Train a custom YOLOv8 model

The `train.py` script is a light wrapper around Ultralytics training.

### 4.1 Prepare your dataset

Your dataset should be in **YOLO format**:

- `images/train`, `images/val`
- `labels/train`, `labels/val`

Create a YAML file, e.g. `data/dataset.yaml`:

```yaml
path: ../  # root directory of your dataset relative to this yaml
train: images/train
val: images/val

names:
  0: class0
  1: class1
```

> Adjust `path`, `train`, `val`, and `names` according to your dataset.

Place this YAML file in `data/dataset.yaml` inside the project (you can change the path with `--data`).

### 4.2 Run training

```bash
cd yolov8_project
.\yolo_env\Scripts\Activate.ps1   # if not already active

python train.py --data data/dataset.yaml --epochs 100 --imgsz 640 --batch 16
```

Main options:

- `--model`: YOLOv8 model variant (default: `yolov8n.pt`).
- `--data`: Path to dataset YAML.
- `--epochs`: Number of training epochs.
- `--imgsz`: Image size (default 640).
- `--batch`: Batch size.

Training outputs (weights, metrics) will be stored in `runs/detect` or `runs/train` folders created by Ultralytics.

---

## 5. Project Structure

```
yolov8_project/
├── yolo_env/              # Virtual environment
├── requirements.txt       # Python dependencies
├── yolov8n.pt           # Pre-trained YOLOv8 nano model
├── detect.py            # General detection script (images/videos/webcam)
├── webcam_detect.py     # Dedicated webcam detection script
├── train.py             # Training script for custom models
├── README.md            # This documentation
└── runs/                # Output directory (auto-created)
    ├── detect/          # Detection results
    └── train/           # Training results
```

---

## 6. Files in this project

- `requirements.txt` – Python dependencies (ultralytics, opencv-python, numpy).
- `detect.py` – Run YOLOv8 inference on images/videos/webcam with full options.
- `webcam_detect.py` – Simplified real-time webcam detection.
- `train.py` – Train a YOLOv8 model on a custom dataset.
- `yolov8n.pt` – Pre-trained YOLOv8 nano model weights.
- `README.md` – This comprehensive guide.

---

## 7. Troubleshooting

**Common Issues:**

1. **ModuleNotFoundError**: Make sure virtual environment is activated
   ```bash
   .\yolo_env\Scripts\Activate.ps1
   ```

2. **PyTorch installation issues**: Install PyTorch first from https://pytorch.org/get-started/locally/

3. **Webcam not working**: Check webcam index (try `--device 1` instead of default)

4. **No detections shown**: Add `--show` flag to see results in window

**Performance Tips:**
- Lower confidence threshold: `--conf 0.1` (detects more objects)
- Use smaller model: `--weights yolov8n.pt` (faster)
- Reduce image size: `--imgsz 320` (faster processing)

---

## 8. Next Steps

You can now customize the project by:
- Training on your own dataset with custom classes
- Integrating into a GUI application
- Building a web API with Flask/FastAPI
- Adding object tracking functionality
- Deploying to edge devices

Happy object detecting! 🚀

