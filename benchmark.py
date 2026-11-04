"""
benchmark.py - run every model x mode on the SAME video and save results for the paper.
Usage:  python benchmark.py path/to/video.mp4 [max_frames]
Output: benchmark_results.csv
"""
import sys, csv, time, cv2
from detect import MODELS, load_model, get_detections
from navigation import decide

def run(video, max_frames=300):
    rows = []
    for name in MODELS:
        model = load_model(name)
        for mode in ("baseline", "adaptive"):
            cap = cv2.VideoCapture(video)
            n = lat = dets_total = stops = 0
            while n < max_frames:
                ok, f = cap.read()
                if not ok: break
                h, w = f.shape[:2]
                t0 = time.time(); dets = get_detections(model, f); lat += (time.time() - t0) * 1000
                cmd, _ = decide(dets, w, h, mode)
                dets_total += len(dets); stops += cmd == "Stop"; n += 1
            cap.release()
            if n:
                rows.append(dict(model=name, mode=mode, frames=n, avg_latency_ms=round(lat / n, 1),
                                 fps=round(1000 / (lat / n), 1), dets_per_frame=round(dets_total / n, 2),
                                 stop_rate=round(stops / n, 3)))
                print(rows[-1])
    with open("benchmark_results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    print("saved benchmark_results.csv")

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
