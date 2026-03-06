import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 real-time object detection from webcam.")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 weights (e.g. yolov8n.pt, yolov8s.pt, or custom .pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Webcam index as string (default: '0').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Always use webcam as source (int index)
    try:
        cam_index = int(args.device)
    except ValueError:
        cam_index = 0

    model = YOLO(args.weights)

    # Ultralytics handles the capture loop; press 'q' in the window to quit.
    model.predict(
        source=cam_index,
        conf=args.conf,
        show=True,
        save=False,
    )


if __name__ == "__main__":
    main()

