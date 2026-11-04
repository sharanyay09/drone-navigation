"""
navigation.py
Phase 3-4: Obstacle position/distance estimation + rule-based navigation.

These are pure functions - test them with fake bounding boxes,
no camera needed, before connecting to detect.py.
"""

# Tune these against your own footage (see calibration note below).
FRAME_WIDTH = 640          # match your camera resolution
CENTER_MARGIN = 0.20       # +/- 20% of frame width counts as "centered"
CLOSE_HEIGHT_RATIO = 0.45  # box height > 45% of frame height => "close"


def get_obstacle_position(box, frame_width=FRAME_WIDTH):
    """
    Classify a bounding box as 'left', 'center', or 'right'
    based on where its center-x falls relative to the frame.
    """
    x1, _, x2, _ = box
    box_center_x = (x1 + x2) / 2
    frame_center_x = frame_width / 2
    margin_px = frame_width * CENTER_MARGIN

    if box_center_x < frame_center_x - margin_px:
        return "left"
    elif box_center_x > frame_center_x + margin_px:
        return "right"
    else:
        return "center"


def is_close(box, frame_height):
    """
    Approximate distance using box height as a proxy: a taller box
    (relative to frame height) means the obstacle is closer.
    """
    _, y1, _, y2 = box
    box_height = y2 - y1
    return (box_height / frame_height) > CLOSE_HEIGHT_RATIO


def pick_priority_obstacle(detections, frame_height):
    """
    If multiple obstacles are detected, the 'priority' one is the
    closest (largest box) - that's the one most urgent to react to.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: (d["box"][3] - d["box"][1]))


def decide_navigation_command(detections, frame_width=FRAME_WIDTH, frame_height=480):
    """
    Core decision logic (Phase 4):
      - No obstacle              -> Move Forward
      - Obstacle centered + close -> Stop
      - Obstacle left             -> Move Right
      - Obstacle right            -> Move Left
      - Obstacle centered, not close -> Move Forward (not urgent yet)
    """
    obstacle = pick_priority_obstacle(detections, frame_height)
    if obstacle is None:
        return "Move Forward"

    position = get_obstacle_position(obstacle["box"], frame_width)
    close = is_close(obstacle["box"], frame_height)

    if position == "center" and close:
        return "Stop"
    elif position == "left":
        return "Move Right"
    elif position == "right":
        return "Move Left"
    else:
        return "Move Forward"


# ---------------------------------------------------------------------------
# Adaptive risk-based navigation (research contribution)
# Compares against decide_navigation_command() above as the baseline.
# ---------------------------------------------------------------------------
WARN_RISK = 0.25   # centre-zone risk above this -> steer away
STOP_RISK = 0.60   # centre-zone risk above this -> stop
CENTER_WEIGHT = 1.5  # obstacles in the flight path count more


def assess_risk(detections, frame_width=FRAME_WIDTH, frame_height=480):
    """
    risk = confidence x proximity x (centre weight).
    Proximity = box height / frame height (same proxy as is_close()), capped at 1.
    Annotates each detection with zone/prox/risk and returns summed risk per zone.
    """
    zones = {"left": 0.0, "center": 0.0, "right": 0.0}
    for d in detections:
        _, y1, _, y2 = d["box"]
        d["zone"] = get_obstacle_position(d["box"], frame_width)
        d["prox"] = round(min(1.0, (y2 - y1) / frame_height / 0.8), 2)
        w = CENTER_WEIGHT if d["zone"] == "center" else 1.0
        d["risk"] = round(d["conf"] * d["prox"] * w, 3)
        zones[d["zone"]] += d["risk"]
    return zones


def decide_adaptive_command(detections, frame_width=FRAME_WIDTH, frame_height=480,
                            warn=WARN_RISK, stop=STOP_RISK):
    """Stop / steer toward the freer side / go forward, based on summed centre risk."""
    zones = assess_risk(detections, frame_width, frame_height)
    if zones["center"] >= stop:
        return "Stop"
    if zones["center"] >= warn:
        return "Move Left" if zones["left"] <= zones["right"] else "Move Right"
    return "Move Forward"


def decide(detections, frame_width, frame_height, mode="adaptive", warn=WARN_RISK, stop=STOP_RISK):
    """Single entry point: returns (command, zone_risks). mode = 'adaptive' | 'baseline'."""
    zones = assess_risk(detections, frame_width, frame_height)  # also annotates boxes
    if mode == "baseline":
        return decide_navigation_command(detections, frame_width, frame_height), zones
    return decide_adaptive_command(detections, frame_width, frame_height, warn, stop), zones


if __name__ == "__main__":
    # Quick sanity tests with fake boxes - no camera required.
    test_cases = [
        ([], "Move Forward"),
        ([{"box": (250, 100, 400, 400), "label": "chair", "conf": 0.9}], "Stop"),      # centered, close
        ([{"box": (0, 100, 150, 250), "label": "chair", "conf": 0.9}], "Move Right"),   # left
        ([{"box": (500, 100, 640, 250), "label": "person", "conf": 0.9}], "Move Left"), # right
    ]

    for detections, expected in test_cases:
        result = decide_navigation_command(detections)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] detections={len(detections)} -> {result} (expected {expected})")

    print("--- adaptive ---")
    adaptive_cases = [
        ([], "Move Forward"),
        ([{"box": (250, 100, 400, 400), "label": "chair", "conf": 0.9}], "Stop"),
        ([{"box": (250, 200, 400, 330), "label": "chair", "conf": 0.9}], "Move Left"),   # mid-range, centred
        ([{"box": (0, 100, 150, 250), "label": "chair", "conf": 0.9}], "Move Forward"),  # off-path
    ]
    for detections, expected in adaptive_cases:
        result = decide_adaptive_command(detections)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] detections={len(detections)} -> {result} (expected {expected})")
