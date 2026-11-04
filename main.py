"""
main.py
Phase 5: End-to-end pipeline - camera -> detection -> navigation -> display.

This is the "simulation" demo: it doesn't fly a real drone, it shows
the full decision loop running live and overlays the command on screen,
which is exactly what you'll record for your demonstration video.
"""

import argparse
import cv2

from detect import load_model, get_detections, draw_detections
from navigation import decide
from utils import overlay_command, log_frame_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n")
    ap.add_argument("--mode", default="adaptive", choices=["adaptive", "baseline"])
    args = ap.parse_args()
    model = load_model(args.model)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame size: {frame_width}x{frame_height}. Press 'q' to quit.")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = get_detections(model, frame)
        command, _ = decide(detections, frame_width, frame_height, args.mode)

        frame = draw_detections(frame, detections)
        frame = overlay_command(frame, command)
        log_frame_result(frame_index, detections, command)

        cv2.imshow("Drone Navigation - Full Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
