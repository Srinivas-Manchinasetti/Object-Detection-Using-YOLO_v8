import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 object detection (image/video/webcam).")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 weights (e.g. yolov8n.pt, yolov8s.pt, or custom .pt).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source: path to image/video/dir or webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show results in a window (OpenCV).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results (images/videos) to runs/detect/...",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load model
    model = YOLO(args.weights)

    # Determine correct source type for YOLO (int for webcam, string for files/dirs)
    source: int | str
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    # Run prediction
    results = model.predict(
        source=source,
        conf=args.conf,
        show=args.show,
        save=args.save,
    )

    # Simple console summary
    for i, res in enumerate(results):
        path = Path(getattr(res, "path", f"frame_{i}"))
        boxes = getattr(res, "boxes", None)
        num_det = int(boxes.shape[0]) if boxes is not None else 0  # type: ignore[call-arg]
        print(f"[{i}] {path}: {num_det} detections")


if __name__ == "__main__":
    main()

