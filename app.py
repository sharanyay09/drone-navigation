"""Web dashboard. Reuses detect.py / navigation.py / utils.py.  Run: python app.py -> http://localhost:5000"""
import os, time, sqlite3, cv2, psutil
import numpy as np
from flask import Flask, Response, jsonify, request, render_template
from detect import MODELS, load_model, get_detections
from navigation import decide
import navigation as nav

os.makedirs("uploads", exist_ok=True)
app = Flask(__name__)
DB, _loaded, BENCH = "drone.db", {}, {}
S = {"model": "yolov8n", "mode": "adaptive", "source": 0, "decision": "Move Forward",
     "zones": {"left": 0, "center": 0, "right": 0}, "dets": [], "fps": 0, "latency": 0, "cpu": 0,
     "total": 0, "alerts": 0, "series": [], "cfg": {"warn": nav.WARN_RISK, "stop": nav.STOP_RISK, "conf": 0.5}}
COL = {"Stop": (0, 0, 255), "Move Left": (255, 200, 0), "Move Right": (255, 200, 0), "Move Forward": (0, 255, 0)}

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
with db() as c:
    c.execute("CREATE TABLE IF NOT EXISTS log(id INTEGER PRIMARY KEY, ts TEXT, model TEXT, decision TEXT, n INT, risk REAL, latency REAL)")

def get_model(name):
    if name not in _loaded: _loaded[name] = load_model(name)
    return _loaded[name]

ST = {"prev": "Move Forward", "last_log": 0, "last_t": time.time()}

def process(f):
    """Detect + decide + annotate one frame. Used by both the video stream and the browser camera."""
    name, mode, cfg = S["model"], S["mode"], S["cfg"]
    h, w = f.shape[:2]
    t0 = time.time(); dets = get_detections(get_model(name), f, cfg["conf"]); lat = (time.time() - t0) * 1000
    cmd, z = decide(dets, w, h, mode, cfg["warn"], cfg["stop"])
    for d in dets:
        x1, y1, x2, y2 = d["box"]; c = (0, 0, 255) if d["risk"] > cfg["warn"] else (0, 200, 255) if d["risk"] > .1 else (80, 220, 80)
        cv2.rectangle(f, (x1, y1), (x2, y2), c, 2)
        cv2.putText(f, f'{d["label"]} {d["conf"]:.2f}', (x1, max(15, y1 - 6)), 0, .5, c, 2)
    cv2.putText(f, cmd, (20, 40), 0, 1.0, COL[cmd], 3)
    now = time.time(); S["fps"] = round(S["fps"] * .8 + .2 / max(now - ST["last_t"], 1e-3), 1); ST["last_t"] = now
    S.update(latency=round(lat, 1), decision=cmd, zones={k: round(v, 2) for k, v in z.items()},
             dets=[{k: v for k, v in d.items() if k != "box"} for d in dets])
    S["total"] += len(dets); S["alerts"] += (cmd != "Move Forward" and ST["prev"] == "Move Forward"); ST["prev"] = cmd
    b = BENCH.setdefault(f"{name} | {mode}", dict(frames=0, lat=0, fps=0, conf=0, dets=0, stops=0))
    b["frames"] += 1; b["lat"] += lat; b["fps"] += S["fps"]; b["dets"] += len(dets)
    b["conf"] += sum(d["conf"] for d in dets); b["stops"] += cmd == "Stop"
    if now - ST["last_log"] >= 1:
        ST["last_log"] = now; S["cpu"] = psutil.cpu_percent()
        S["series"] = (S["series"] + [{"fps": S["fps"], "lat": S["latency"], "cpu": S["cpu"]}])[-60:]
        with db() as c:
            c.execute("INSERT INTO log(ts,model,decision,n,risk,latency) VALUES(?,?,?,?,?,?)",
                      (time.strftime("%H:%M:%S"), f"{name} | {mode}", cmd, len(dets), z["center"], lat))
    return f

def stream():
    """Server-side source (uploaded video file)."""
    cap, cur = None, None
    while True:
        if S["source"] != cur:
            if cap: cap.release()
            cap, cur = cv2.VideoCapture(S["source"]), S["source"]
        ok, f = cap.read()
        if not ok:
            if isinstance(cur, str): cap.set(cv2.CAP_PROP_POS_FRAMES, 0); time.sleep(.05); continue
            return  # no server camera (cloud) - end stream instead of spinning
        yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + cv2.imencode(".jpg", process(f))[1].tobytes() + b"\r\n"

@app.route("/api/frame", methods=["POST"])
def frame():
    """Browser camera: receives one JPEG, returns it annotated."""
    f = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if f is None: return "bad frame", 400
    return Response(cv2.imencode(".jpg", process(f))[1].tobytes(), mimetype="image/jpeg")

@app.route("/")
def index(): return render_template("index.html", models=list(MODELS))
@app.route("/video")
def video(): return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=f")
@app.route("/api/stats")
def stats(): return jsonify(S)
@app.route("/api/history")
def history():
    with db() as c: return jsonify([dict(r) for r in c.execute("SELECT * FROM log ORDER BY id DESC LIMIT 100")])
@app.route("/api/models")
def models():
    return jsonify({k: {"frames": b["frames"], "latency": round(b["lat"] / b["frames"], 1), "fps": round(b["fps"] / b["frames"], 1),
                        "avg_conf": round(b["conf"] / max(b["dets"], 1), 2), "dets_per_frame": round(b["dets"] / b["frames"], 2),
                        "stop_rate": round(b["stops"] / b["frames"], 3)} for k, b in BENCH.items() if b["frames"]})
@app.route("/api/model", methods=["POST"])
def set_model():
    if request.json.get("model") in MODELS: S["model"] = request.json["model"]
    return jsonify(model=S["model"])
@app.route("/api/mode", methods=["POST"])
def set_mode():
    if request.json.get("mode") in ("adaptive", "baseline"): S["mode"] = request.json["mode"]
    return jsonify(mode=S["mode"])
@app.route("/api/config", methods=["POST"])
def config():
    for k in S["cfg"]:
        if k in request.json: S["cfg"][k] = float(request.json[k])
    return jsonify(S["cfg"])
@app.route("/api/source", methods=["POST"])
def source():
    if "file" in request.files:
        p = os.path.join("uploads", request.files["file"].filename); request.files["file"].save(p); S["source"] = p
    else: S["source"] = 0
    return jsonify(source=str(S["source"]))

if __name__ == "__main__": app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
