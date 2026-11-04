"""
utils.py
Shared helper functions used by detect.py and navigation.py.
"""

import cv2


def overlay_command(frame, command: str):
    """Draw the current navigation command as text on the frame."""
    color_map = {
        "Stop": (0, 0, 255),
        "Move Left": (255, 200, 0),
        "Move Right": (255, 200, 0),
        "Move Forward": (0, 255, 0),
    }
    color = color_map.get(command, (255, 255, 255))
    cv2.putText(frame, command, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 3)
    return frame


def log_frame_result(frame_index: int, detections, command: str):
    """Print a one-line log for each processed frame (useful for the evaluation report)."""
    labels = [d["label"] for d in detections]
    print(f"Frame {frame_index:05d} | Obstacles: {labels} | Command: {command}")
