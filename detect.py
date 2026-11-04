"""
detect.py
Phase 1-2: Video capture + YOLOv8 object detection.

Run this file directly to test detection on your webcam before
wiring it into navigation.py.
"""

import cv2
from ultralytics import YOLO, RTDETR

# Nano model = fastest, good enough for real-time CPU inference.
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.5


# Registry used by the dashboard / experiments: key -> (weights, class)
MODELS = {
    "yolov8n": ("yolov8n.pt", YOLO),
    "yolov10n": ("yolov10n.pt", YOLO),
    "yolo11n": ("yolo11n.pt", YOLO),
    "rtdetr-l": ("rtdetr-l.pt", RTDETR),
}


def load_model(name: str = MODEL_PATH):
    """Load a model by registry key ("yolov8n", "rtdetr-l", ...) or by weights path."""
    key = name.replace(".pt", "")
    if key in MODELS:
        weights, cls = MODELS[key]
        return cls(weights)
    return YOLO(name)


def get_detections(model, frame, conf_threshold: float = CONF_THRESHOLD):
    """
    Run YOLOv8 inference on a single frame.

    Returns a list of dicts: [{"box": (x1,y1,x2,y2), "label": str, "conf": float}, ...]
    """
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        detections.append({"box": (x1, y1, x2, y2), "label": label, "conf": conf})

    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes + labels on the frame (for visual debugging)."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f'{det["label"]} {det["conf"]:.2f}'
        cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return frame


def main():
    """Standalone test: webcam -> YOLOv8 -> annotated video window."""
    model = load_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam. Check your camera index/permissions.")
        return

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = get_detections(model, frame)
        frame = draw_detections(frame, detections)

        cv2.imshow("Detection - Phase 2 Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
