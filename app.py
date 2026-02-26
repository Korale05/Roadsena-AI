"""
RoadSense AI — Complete Flask Backend
All routes, AI detection, scoring, PDF, WhatsApp, Chat Agent in one file.
"""

import os, json, base64, uuid, time, re
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, send_file, Response)
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─── App Setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "roadsense_secret_2024"

# Folders
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ─── Load YOLO Models ────────────────────────────────────────────────────────
# Model 1 — Your high-accuracy pothole/segmentation model (REQUIRED)
MODEL1_PATH = r"D:\Pathhole_Detection\YOLOv8_Pothole_Segmentation_Road_Damage_Assessment\model\best.pt"

# Model 2 — Crack detection model (OPTIONAL).
# Point this to a crack-specific .pt file if you have one.
# Set to None to run in single-model mode — everything still works perfectly.
# Recommended: download a YOLOv8 trained on RDD2022 (Road Damage Detection 2022)
# which covers D00 Longitudinal Crack, D10 Transverse Crack, D20 Alligator Crack.
MODEL2_PATH = None   # e.g. r"D:\Pathhole_Detection\models\rdd2022_crack.pt"

import torch
import torch.serialization

def _patch_pytorch_26():
    """PyTorch 2.6 changed weights_only=True by default.
    Ultralytics uses custom classes that must be allowlisted before YOLO loads."""
    try:
        from ultralytics.nn.tasks import (
            DetectionModel, SegmentationModel, PoseModel, ClassificationModel
        )
        torch.serialization.add_safe_globals([
            DetectionModel, SegmentationModel, PoseModel, ClassificationModel
        ])
        print("[RoadSense] PyTorch 2.6 safe-globals patch applied ✓")
    except AttributeError:
        pass  # Older PyTorch — no patch needed
    except ImportError:
        pass  # Classes not present in this ultralytics version

_patch_pytorch_26()

print(f"[RoadSense] Loading Model 1 (Pothole) ...")
model1 = YOLO(MODEL1_PATH)
print(f"[RoadSense] Model 1 loaded ✓  classes: {list(model1.names.values())}")

model2 = None
if MODEL2_PATH and os.path.exists(MODEL2_PATH):
    print(f"[RoadSense] Loading Model 2 (Crack) ...")
    model2 = YOLO(MODEL2_PATH)
    print(f"[RoadSense] Model 2 loaded ✓  classes: {list(model2.names.values())}")
else:
    print("[RoadSense] Model 2 not configured — single-model mode")

model = model1   # backward-compat alias

# ─── In-Memory Database ──────────────────────────────────────────────────────
defects_db      = []  # Individual frame detections (internal use)
road_records_db = []  # ONE record per road per session (shown on map/queue/history)
sessions_db     = {}  # Active live-inspection sessions keyed by session_id
officers_db     = {   # Hard-coded officers (no DB needed for hackathon)
    "admin":   {"password": "admin123",   "name": "Admin Officer",    "phone": "+91XXXXXXXXXX"},
    "officer1":{"password": "officer123", "name": "Field Officer 1",  "phone": "+91XXXXXXXXXX"},
    "officer2":{"password": "officer456", "name": "Field Officer 2",  "phone": "+91XXXXXXXXXX"},
}

# ─── Damage Constants ────────────────────────────────────────────────────────
DAMAGE_CONFIG = {
    "pothole":            {"base": 9, "label": "Pothole",            "repair_cost": 25000},
    "D40":                {"base": 9, "label": "Pothole",            "repair_cost": 25000},
    "alligator_crack":    {"base": 7, "label": "Alligator Crack",    "repair_cost": 15000},
    "D20":                {"base": 7, "label": "Alligator Crack",    "repair_cost": 15000},
    "transverse_crack":   {"base": 5, "label": "Transverse Crack",   "repair_cost":  8000},
    "D10":                {"base": 5, "label": "Transverse Crack",   "repair_cost":  8000},
    "longitudinal_crack": {"base": 3, "label": "Longitudinal Crack", "repair_cost":  5000},
    "D00":                {"base": 3, "label": "Longitudinal Crack", "repair_cost":  5000},
}

SEVERITY_LEVELS = [
    (0, 2,  "Good",      "success", "#28a745", 90),
    (2, 4,  "Moderate",  "warning", "#ffc107", 70),
    (4, 6,  "Severe",    "orange",  "#fd7e14", 50),
    (6, 8,  "Critical",  "danger",  "#dc3545", 30),
    (8, 10, "Emergency", "dark",    "#7b0000", 10),
]

# ─── Helper Functions ─────────────────────────────────────────────────────────

def get_level(score):
    for lo, hi, label, bs_class, color, _ in SEVERITY_LEVELS:
        if lo <= score <= hi:
            return label, bs_class, color
    return "Emergency", "dark", "#7b0000"

def get_repair_days(score):
    if score >= 8: return "Immediately — Today"
    if score >= 6: return "Within 7 days"
    if score >= 4: return "Within 30 days"
    if score >= 2: return "Within 90 days"
    return "Monitor monthly"

def calc_economic_impact(score):
    return round(score * 150 * 500 * 30)

def calc_score(detections, img_w, img_h):
    """Core scoring engine matching the documentation."""
    if not detections:
        return 0.0, []

    img_area = img_w * img_h
    scored = []

    for det in detections:
        cls_name = det["class"].lower().replace(" ", "_")
        cfg = DAMAGE_CONFIG.get(cls_name, {"base": 5, "label": cls_name, "repair_cost": 10000})
        base = cfg["base"]

        # Size ratio
        bx1, by1, bx2, by2 = det["bbox"]
        box_area = (bx2 - bx1) * (by2 - by1)
        size_ratio = box_area / img_area if img_area > 0 else 0
        if   size_ratio > 0.06: size_mult = 1.6
        elif size_ratio > 0.02: size_mult = 1.3
        else:                   size_mult = 1.0

        # Confidence weight
        conf = det.get("confidence", 0.8)
        if   conf >= 0.90: conf_w = 1.00
        elif conf >= 0.70: conf_w = 0.85
        elif conf >= 0.50: conf_w = 0.70
        else:              conf_w = 0.55

        single_score = min(10.0, base * size_mult * conf_w)
        scored.append({**det, "single_score": round(single_score, 2),
                       "label": cfg["label"], "repair_cost": cfg["repair_cost"]})

    n = len(scored)
    avg = sum(d["single_score"] for d in scored) / n
    count_penalty = 1 + min(0.4, (n - 1) * 0.10)
    final = min(10.0, avg * count_penalty)
    return round(final, 2), scored

def annotate_image(image_bytes, detections):
    """Draw bounding boxes on image, return annotated bytes."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    COLOR_MAP = {
        "Pothole":            (0,   0,   200),
        "Alligator Crack":    (0,   165, 255),
        "Transverse Crack":   (0,   200, 200),
        "Longitudinal Crack": (0,   200, 0),
    }

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det.get("label", det["class"])
        conf  = det.get("confidence", 0)
        color = COLOR_MAP.get(label, (100, 100, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

def _extract_boxes(yolo_model, img_cv2):
    """Run one YOLO model on a decoded cv2 image, return raw detection dicts."""
    results = yolo_model(img_cv2, verbose=False)[0]
    dets = []
    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        conf     = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        dets.append({"class": cls_name, "confidence": round(conf, 3), "bbox": [x1,y1,x2,y2]})
    return dets

def _bbox_iou(a, b):
    """IoU of two [x1,y1,x2,y2] boxes — used for duplicate suppression."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def run_yolo(image_bytes):
    """Run Model1 (pothole) + optional Model2 (crack) on image bytes.
    Both models see the same frame. Results are merged with IoU-based
    duplicate suppression so the same damage is not double-counted.
    Extra time for Model2 is ~0.2-0.3s — well within the 2-second capture window.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return [], 640, 480

    h, w = img.shape[:2]

    # Model 1 always runs (your high-accuracy pothole model)
    detections = _extract_boxes(model1, img)

    # Model 2 runs only when configured
    if model2 is not None:
        for cd in _extract_boxes(model2, img):
            # Skip if already detected by Model1 (IoU > 0.4 = same damage)
            if not any(_bbox_iou(d["bbox"], cd["bbox"]) > 0.4 for d in detections):
                detections.append(cd)

    return detections, w, h

def _save_image(annotated_bytes):
    """Save annotated image bytes to disk, return URL."""
    if not annotated_bytes:
        return None
    fname = f"{uuid.uuid4().hex}.jpg"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "wb") as fh:
        fh.write(annotated_bytes)
    return f"/static/outputs/{fname}"

def save_frame_to_session(session_id, lat, lng, detections, score, annotated_bytes):
    """During a live inspection each frame is buffered inside the session.
    Nothing is written to road_records_db yet — that happens on Stop."""
    sess = sessions_db.get(session_id)
    if not sess:
        return
    img_url = _save_image(annotated_bytes)
    sess["frames"].append({
        "lat": lat, "lng": lng,
        "detections": detections,
        "score": score,
        "img_url": img_url,
        "timestamp": datetime.now().isoformat(),
    })
    # Track the worst frame for quick access
    if score > sess["worst_score"]:
        sess["worst_score"] = score
        sess["worst_img"]   = img_url
        sess["worst_lat"]   = lat
        sess["worst_lng"]   = lng

def finalise_session(session_id):
    """Called when officer clicks Stop.
    Aggregates all frames into ONE road record and saves to road_records_db.
    Map shows this single record — not 50 individual dots."""
    sess = sessions_db.get(session_id)
    if not sess:
        return None

    frames = sess.get("frames", [])
    road_name = sess["road_name"]

    if not frames:
        # No detections on this road — save a clean record
        level, bs_class, color = get_level(0)
        record = {
            "id": str(uuid.uuid4()), "road_name": road_name,
            "lat": sess.get("start_lat", 0), "lng": sess.get("start_lng", 0),
            "inspection_date": sess["start_time"][:10],
            "inspection_time": sess["start_time"][11:19],
            "timestamp": sess["start_time"],
            "frames_captured": sess["frames_captured"],
            "defect_frames": 0,
            "all_detections": [],
            "score": 0, "level": "Good", "bs_class": "success", "color": "#28a745",
            "annotated_img": None, "worst_img": None,
            "economic_impact": 0, "repair_action": "Monitor monthly",
            "source": "webcam", "status": "pending",
            "officer": sess.get("officer", ""),
            "route_points": [],
        }
    else:
        # Aggregate: overall score = average of all defect-frame scores
        scores   = [f["score"] for f in frames]
        avg_score = round(sum(scores) / len(scores), 2)
        worst     = sess["worst_score"]
        # Weighted: 60% worst + 40% average — captures both severity and frequency
        final_score = round(min(10.0, worst * 0.6 + avg_score * 0.4), 2)

        # Collect all unique detection types across all frames
        all_types = {}
        for fr in frames:
            for d in fr["detections"]:
                lbl = d.get("label", d.get("class", "Unknown"))
                all_types[lbl] = all_types.get(lbl, 0) + 1

        level, bs_class, color = get_level(final_score)
        route_points = [{"lat": f["lat"], "lng": f["lng"], "score": f["score"]} for f in frames]

        record = {
            "id": str(uuid.uuid4()), "road_name": road_name,
            # Use worst-detection GPS as the map marker location
            "lat": sess["worst_lat"], "lng": sess["worst_lng"],
            "inspection_date": sess["start_time"][:10],
            "inspection_time": sess["start_time"][11:19],
            "timestamp": sess["start_time"],
            "frames_captured": sess["frames_captured"],
            "defect_frames": len(frames),
            "all_detections": [{"label": k, "count": v} for k, v in all_types.items()],
            "score": final_score, "level": level, "bs_class": bs_class, "color": color,
            "annotated_img": sess["worst_img"],
            "worst_img": sess["worst_img"],
            "economic_impact": calc_economic_impact(final_score),
            "repair_action":   get_repair_days(final_score),
            "source": "webcam", "status": "pending",
            "officer": sess.get("officer", ""),
            "route_points": route_points,
        }

    road_records_db.append(record)
    # Also mirror into defects_db so existing API routes still work
    defects_db.append(record)
    del sessions_db[session_id]
    return record

def save_record(road_name, lat, lng, detections, score, annotated_bytes, source="citizen"):
    """Direct save for citizen uploads, video batch, demo data.
    Creates one road_record immediately (no session buffering needed)."""
    img_url = _save_image(annotated_bytes)
    level, bs_class, color = get_level(score)
    all_types = {}
    for d in detections:
        lbl = d.get("label", d.get("class","Unknown"))
        all_types[lbl] = all_types.get(lbl, 0) + 1
    record = {
        "id": str(uuid.uuid4()), "road_name": road_name,
        "lat": lat, "lng": lng,
        "inspection_date": datetime.now().strftime("%Y-%m-%d"),
        "inspection_time": datetime.now().strftime("%H:%M:%S"),
        "timestamp": datetime.now().isoformat(),
        "frames_captured": 1, "defect_frames": 1 if detections else 0,
        "all_detections": [{"label": k, "count": v} for k, v in all_types.items()],
        "detections": detections,   # kept for backward compat
        "score": score, "level": level, "bs_class": bs_class, "color": color,
        "annotated_img": img_url, "worst_img": img_url,
        "economic_impact": calc_economic_impact(score),
        "repair_action":   get_repair_days(score),
        "source": source, "status": "pending", "officer": "",
        "route_points": [{"lat": lat, "lng": lng, "score": score}],
    }
    road_records_db.append(record)
    defects_db.append(record)
    return record

def send_whatsapp_alert(record):
    """Send WhatsApp alert via Twilio when score >= 7."""
    try:
        from twilio.rest import Client
        account_sid = os.environ.get("TWILIO_SID", "")
        auth_token  = os.environ.get("TWILIO_TOKEN", "")
        from_number = os.environ.get("TWILIO_FROM", "whatsapp:+14155238886")
        to_number   = os.environ.get("TWILIO_TO",   "whatsapp:+91XXXXXXXXXX")

        if not account_sid or not auth_token:
            return  # Skip if not configured

        client = Client(account_sid, auth_token)
        msg = (f"🚨 RoadSense ALERT\n"
               f"Road: {record['road_name']}\n"
               f"Score: {record['score']}/10 — {record['level']}\n"
               f"Action: {record['repair_action']}\n"
               f"Economic Risk: ₹{record['economic_impact']:,}/30 days\n"
               f"GPS: {record['lat']}, {record['lng']}")
        client.messages.create(body=msg, from_=from_number, to=to_number)
    except Exception as e:
        print(f"[WhatsApp] Failed: {e}")

def get_city_health_score():
    if not road_records_db:
        return 100
    avg_severity = sum(r["score"] for r in road_records_db) / len(road_records_db)
    rqi = 100 - (avg_severity * 10)
    return round(max(0, min(100, rqi)), 1)

def get_stats():
    total   = len(road_records_db)
    crit    = sum(1 for r in road_records_db if r["score"] >= 6)
    emerg   = sum(1 for r in road_records_db if r["score"] >= 8)
    eco     = sum(r["economic_impact"] for r in road_records_db)
    city_hs = get_city_health_score()
    return {"total": total, "critical": crit, "emergency": emerg,
            "economic_risk": eco, "city_health": city_hs}

# ─── Auth Helpers ─────────────────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "officer" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Auth ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "officer" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        uname = request.form.get("username", "").strip()
        pwd   = request.form.get("password", "").strip()
        if uname in officers_db and officers_db[uname]["password"] == pwd:
            session["officer"] = uname
            session["name"]    = officers_db[uname]["name"]
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─── Dashboard ────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", officer=session.get("name"))

# ─── City Map ────────────────────────────────────────────────────────────────
@app.route("/city-map")
@login_required
def city_map():
    return render_template("city_map.html", officer=session.get("name"))

# ─── Repair Queue ─────────────────────────────────────────────────────────────
@app.route("/repair-queue")
@login_required
def repair_queue():
    return render_template("repair_queue.html", officer=session.get("name"))

# ─── Inspection History ───────────────────────────────────────────────────────
@app.route("/history")
@login_required
def history():
    return render_template("history.html", officer=session.get("name"))

# ─── Live Webcam Inspection ───────────────────────────────────────────────────
@app.route("/live-inspection")
@login_required
def live_inspection():
    return render_template("live_inspection.html", officer=session.get("name"))

# ─── Video File Upload ────────────────────────────────────────────────────────
@app.route("/video-upload")
@login_required
def video_upload():
    return render_template("video_upload.html", officer=session.get("name"))

# ─── Citizen Report (Public — No Login) ──────────────────────────────────────
@app.route("/report")
def citizen_report():
    return render_template("citizen_report.html")

# ═══════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Stats for Dashboard ──────────────────────────────────────────────────────
@app.route("/api/stats")
@login_required
def api_stats():
    s = get_stats()
    # Severity distribution
    dist = {"Good": 0, "Moderate": 0, "Severe": 0, "Critical": 0, "Emergency": 0}
    for r in road_records_db:
        dist[r["level"]] = dist.get(r["level"], 0) + 1
    # Damage type distribution
    dtype_dist = {}
    for r in road_records_db:
        for d in r.get("all_detections", r.get("detections", [])):
            lbl = d.get("label", d.get("class", "Unknown"))
            cnt = d.get("count", 1)
            dtype_dist[lbl] = dtype_dist.get(lbl, 0) + cnt
    # Recent 5
    recent = sorted(road_records_db, key=lambda x: x["timestamp"], reverse=True)[:5]
    return jsonify({**s, "severity_dist": dist, "damage_types": dtype_dist,
                    "recent": recent})

# ─── City Health Score ────────────────────────────────────────────────────────
@app.route("/api/city-health-score")
def api_city_health():
    return jsonify({"score": get_city_health_score(), "total": len(defects_db)})

# ─── Heatmap Data for Map ─────────────────────────────────────────────────────
@app.route("/api/heatmap-data")
def api_heatmap():
    data = []
    for r in road_records_db:
        if r.get("lat") and r.get("lng"):
            data.append({
                "id": r["id"], "road_name": r["road_name"],
                "lat": r["lat"], "lng": r["lng"],
                "score": r["score"], "level": r["level"],
                "color": r["color"], "timestamp": r["timestamp"],
                "inspection_date": r.get("inspection_date",""),
                "inspection_time": r.get("inspection_time",""),
                "annotated_img": r.get("annotated_img") or r.get("worst_img"),
                "economic_impact": r["economic_impact"],
                "repair_action": r["repair_action"],
                "defect_frames": r.get("defect_frames", 0),
                "frames_captured": r.get("frames_captured", 1),
                "all_detections": r.get("all_detections", []),
                "officer": r.get("officer",""),
                "source": r.get("source", ""),
                "route_points": r.get("route_points", []),
            })
    return jsonify(data)

# ─── Priority Queue ───────────────────────────────────────────────────────────
@app.route("/api/priority-queue")
@login_required
def api_priority_queue():
    # Group by road_name, take worst score
    roads = {}
    for r in road_records_db:
        rn = r["road_name"]
        if rn not in roads or r["score"] > roads[rn]["score"]:
            roads[rn] = r
    sorted_roads = sorted(roads.values(), key=lambda x: x["score"], reverse=True)
    return jsonify(sorted_roads)

# ─── All Defects (History) ────────────────────────────────────────────────────
@app.route("/api/defects")
@login_required
def api_defects():
    q     = request.args.get("q", "").lower()
    level = request.args.get("level", "")
    data  = road_records_db
    if q:
        data = [r for r in data if q in r["road_name"].lower()]
    if level:
        data = [r for r in data if r["level"].lower() == level.lower()]
    data = sorted(data, key=lambda x: x["timestamp"], reverse=True)
    return jsonify(data)

# ─── Update Status ────────────────────────────────────────────────────────────
@app.route("/api/defects/<record_id>/status", methods=["POST"])
@login_required
def api_update_status(record_id):
    status = request.json.get("status", "pending")
    for r in defects_db:
        if r["id"] == record_id:
            r["status"] = status
            return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Not found"}), 404

# ─── Start Inspection Session ─────────────────────────────────────────────────
@app.route("/api/start-session", methods=["POST"])
@login_required
def api_start_session():
    """Creates a new session. Returns session_id that frontend must send with every frame."""
    data      = request.json
    road_name = data.get("road_name", "Unknown Road").strip()
    lat       = data.get("lat", 0.0)
    lng       = data.get("lng", 0.0)
    sess_id   = uuid.uuid4().hex
    sessions_db[sess_id] = {
        "road_name":      road_name,
        "start_time":     datetime.now().isoformat(),
        "start_lat":      lat,
        "start_lng":      lng,
        "officer":        session.get("name", ""),
        "frames_captured":0,
        "frames":         [],   # only defect frames stored here
        "worst_score":    0,
        "worst_img":      None,
        "worst_lat":      lat,
        "worst_lng":      lng,
    }
    return jsonify({"session_id": sess_id, "road_name": road_name})

# ─── Stop Inspection Session ──────────────────────────────────────────────────
@app.route("/api/stop-session", methods=["POST"])
@login_required
def api_stop_session():
    """Finalises session → aggregates all frames → saves ONE road record."""
    sess_id = request.json.get("session_id", "")
    if sess_id not in sessions_db:
        return jsonify({"error": "Session not found"}), 404
    record = finalise_session(sess_id)
    if not record:
        return jsonify({"error": "Could not finalise session"}), 500
    if record["score"] >= 7:
        send_whatsapp_alert(record)
    return jsonify({
        "road_name":       record["road_name"],
        "score":           record["score"],
        "level":           record["level"],
        "color":           record["color"],
        "frames_captured": record["frames_captured"],
        "defect_frames":   record["defect_frames"],
        "economic_impact": record["economic_impact"],
        "repair_action":   record["repair_action"],
        "annotated_img":   record["annotated_img"],
        "record_id":       record["id"],
        "inspection_date": record["inspection_date"],
        "inspection_time": record["inspection_time"],
        "all_detections":  record["all_detections"],
    })

# ─── Process Single Frame (Live Webcam) ───────────────────────────────────────
@app.route("/api/process-frame", methods=["POST"])
@login_required
def api_process_frame():
    data       = request.json
    sess_id    = data.get("session_id", "")
    lat        = data.get("lat", 0.0)
    lng        = data.get("lng", 0.0)
    img_b64    = data.get("image", "")

    if not img_b64:
        return jsonify({"error": "No image"}), 400

    # Increment frame counter even for clean frames
    sess = sessions_db.get(sess_id)
    if sess:
        sess["frames_captured"] += 1

    # Decode base64 image
    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]
    image_bytes = base64.b64decode(img_b64)

    raw_dets, w, h = run_yolo(image_bytes)
    if not raw_dets:
        # Clean frame — discard, return lightweight response
        return jsonify({"detected": False,
                        "frames_captured": sess["frames_captured"] if sess else 0})

    score, scored_dets = calc_score(raw_dets, w, h)
    annotated          = annotate_image(image_bytes, scored_dets)

    # Buffer inside session — do NOT write to DB yet
    if sess:
        save_frame_to_session(sess_id, lat, lng, scored_dets, score, annotated)

    return jsonify({
        "detected":        True,
        "score":           score,
        "level":           get_level(score)[0],
        "color":           get_level(score)[2],
        "detections":      scored_dets,
        "img_url":         sess["worst_img"] if sess else None,
        "frames_captured": sess["frames_captured"] if sess else 0,
        "defect_frames":   len(sess["frames"]) if sess else 0,
        "worst_score":     sess["worst_score"] if sess else score,
    })

# ─── Process Citizen Photo Upload ─────────────────────────────────────────────
@app.route("/api/upload-image", methods=["POST"])
def api_upload_image():
    road_name = request.form.get("road_name", "Unknown Road")
    lat       = float(request.form.get("lat",  17.6868))
    lng       = float(request.form.get("lng",  75.9079))
    f         = request.files.get("image")

    if not f:
        return jsonify({"error": "No file"}), 400

    image_bytes           = f.read()
    raw_dets, w, h        = run_yolo(image_bytes)
    score, scored_dets    = calc_score(raw_dets, w, h)
    annotated             = annotate_image(image_bytes, scored_dets)
    record                = save_record(road_name, lat, lng, scored_dets, score, annotated, "citizen")

    if score >= 7:
        send_whatsapp_alert(record)

    return jsonify({
        "detected":    len(scored_dets) > 0,
        "score":       score,
        "level":       record["level"],
        "color":       record["color"],
        "detections":  scored_dets,
        "img_url":     record["annotated_img"],
        "record_id":   record["id"],
        "economic":    record["economic_impact"],
        "repair_action": record["repair_action"],
    })

# ─── Process Video File ───────────────────────────────────────────────────────
@app.route("/api/process-video", methods=["POST"])
@login_required
def api_process_video():
    road_name = request.form.get("road_name", "Unknown Road")
    lat       = float(request.form.get("lat",  17.6868))
    lng       = float(request.form.get("lng",  75.9079))
    f         = request.files.get("video")

    if not f:
        return jsonify({"error": "No video file"}), 400

    # Save temp video
    tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{uuid.uuid4().hex}.mp4")
    f.save(tmp_path)

    cap       = cv2.VideoCapture(tmp_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    interval  = int(fps * 2)  # Every 2 seconds
    frame_num = 0
    processed = 0
    results   = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % interval == 0:
                _, buf        = cv2.imencode(".jpg", frame)
                image_bytes   = buf.tobytes()
                raw_dets, w, h = run_yolo(image_bytes)
                if raw_dets:
                    score, scored_dets = calc_score(raw_dets, w, h)
                    annotated          = annotate_image(image_bytes, scored_dets)
                    record             = save_record(road_name, lat, lng, scored_dets,
                                                     score, annotated, "video")
                    if score >= 7:
                        send_whatsapp_alert(record)
                    results.append({"frame": frame_num, "score": score,
                                    "level": record["level"]})
                processed += 1
            frame_num += 1
    finally:
        cap.release()
        os.remove(tmp_path)

    return jsonify({
        "frames_analyzed": processed,
        "defects_found":   len(results),
        "results":         results,
        "city_health":     get_city_health_score(),
    })

# ─── PDF Report ───────────────────────────────────────────────────────────────
@app.route("/api/report/<record_id>")
@login_required
def api_report(record_id):
    record = next((r for r in road_records_db if r["id"] == record_id), None)
    if not record:
        return "Record not found", 404

    pdf_path = os.path.join(REPORT_DIR, f"report_{record_id}.pdf")
    _generate_pdf(record, pdf_path)
    return send_file(pdf_path, as_attachment=True,
                     download_name=f"RoadSense_{record['road_name'].replace(' ','_')}.pdf")

def _generate_pdf(record, pdf_path):
    doc    = SimpleDocTemplate(pdf_path, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # Title
    title_style = ParagraphStyle("title", fontSize=20, fontName="Helvetica-Bold",
                                  textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER,
                                  spaceAfter=6)
    sub_style   = ParagraphStyle("sub",   fontSize=11, fontName="Helvetica",
                                  textColor=colors.HexColor("#666666"),  alignment=TA_CENTER,
                                  spaceAfter=20)
    label_style = ParagraphStyle("label", fontSize=10, fontName="Helvetica-Bold",
                                  textColor=colors.HexColor("#333333"))
    value_style = ParagraphStyle("value", fontSize=10, fontName="Helvetica",
                                  textColor=colors.HexColor("#555555"))
    head_style  = ParagraphStyle("head",  fontSize=13, fontName="Helvetica-Bold",
                                  textColor=colors.HexColor("#1a1a2e"), spaceBefore=16, spaceAfter=8)

    story.append(Paragraph("🛣 RoadSense AI", title_style))
    story.append(Paragraph("Road Inspection Report", sub_style))
    story.append(Spacer(1, 0.3*cm))

    # Metadata table
    ts = record["timestamp"][:19].replace("T", " ")
    meta = [
        ["Road Name", record["road_name"],  "Timestamp", ts],
        ["GPS",       f"{record['lat']}, {record['lng']}", "Source", record.get("source","").capitalize()],
        ["Severity",  f"{record['score']}/10", "Level",  record["level"]],
        ["Action",    record["repair_action"],  "Economic Risk", f"₹{record['economic_impact']:,}"],
    ]
    t = Table(meta, colWidths=[3.5*cm, 6*cm, 3.5*cm, 4.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8f4fd")),
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#e8f4fd")),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",   (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTNAME",   (2,0), (2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",    (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Annotated image
    if record.get("annotated_img"):
        img_path = os.path.join(BASE_DIR, record["annotated_img"].lstrip("/"))
        if os.path.exists(img_path):
            story.append(Paragraph("Annotated Detection Image", head_style))
            story.append(RLImage(img_path, width=15*cm, height=9*cm))
            story.append(Spacer(1, 0.3*cm))

    # Detections detail
    if record.get("detections"):
        story.append(Paragraph("Defect Breakdown", head_style))
        det_data = [["#", "Type", "Confidence", "Score", "Size"]]
        for i, d in enumerate(record["detections"], 1):
            bx = d.get("bbox", [0,0,0,0])
            w  = bx[2]-bx[0]; h = bx[3]-bx[1]
            det_data.append([
                str(i),
                d.get("label", d.get("class","")),
                f"{d.get('confidence',0):.0%}",
                str(d.get("single_score","")),
                f"{int(w)}×{int(h)}px"
            ])
        dt = Table(det_data, colWidths=[1*cm, 5*cm, 3*cm, 3*cm, 4.5*cm])
        dt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",    (0,0), (-1,-1), 6),
        ]))
        story.append(dt)

    # Footer
    story.append(Spacer(1, 1*cm))
    footer_style = ParagraphStyle("foot", fontSize=8, textColor=colors.HexColor("#999999"),
                                   alignment=TA_CENTER)
    story.append(Paragraph(
        f"Generated by RoadSense AI | {datetime.now().strftime('%d %b %Y %H:%M')} | "
        f"Report ID: {record['id'][:8].upper()}", footer_style))

    doc.build(story)

# ─── Budget Optimizer ─────────────────────────────────────────────────────────
@app.route("/api/budget-optimizer", methods=["POST"])
@login_required
def api_budget_optimizer():
    budget = float(request.json.get("budget", 0))
    roads  = {}
    for r in road_records_db:
        rn = r["road_name"]
        if rn not in roads or r["score"] > roads[rn]["score"]:
            roads[rn] = r

    sorted_roads = sorted(roads.values(), key=lambda x: x["score"], reverse=True)

    # Estimate repair cost from detections
    def est_cost(record):
        base_cost = sum(d.get("count", 1) * 10000 for d in record.get("all_detections", record.get("detections",[])))
        return max(base_cost, 5000)

    selected       = []
    remaining      = budget
    old_hs         = get_city_health_score()

    for r in sorted_roads:
        cost = est_cost(r)
        if cost <= remaining:
            selected.append({**r, "estimated_cost": cost})
            remaining -= cost

    # Estimated new health score
    new_hs = old_hs + len(selected) * 2  # Approximate improvement

    return jsonify({
        "selected":       selected,
        "total_roads":    len(selected),
        "budget_used":    budget - remaining,
        "budget_left":    remaining,
        "old_health":     old_hs,
        "estimated_health": round(min(100, new_hs), 1),
    })

# ─── AI Chat Agent ────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    user_msg = request.json.get("message", "")
    if not user_msg:
        return jsonify({"reply": "Please type a message."})

    # Build context from live DB
    stats = get_stats()
    queue = sorted(defects_db, key=lambda x: x["score"], reverse=True)[:10]
    queue_summary = "\n".join(
        f"  - {r['road_name']}: {r['score']}/10 ({r['level']}) — {r['repair_action']} — ₹{r['economic_impact']:,} risk"
        for r in queue
    )

    context = f"""You are RoadSense AI, an intelligent road governance assistant for Indian municipalities.

LIVE DATABASE SUMMARY:
- Total detections: {stats['total']}
- Critical roads (score 6+): {stats['critical']}
- Emergency roads (score 8+): {stats['emergency']}
- Total economic risk (30 days): ₹{stats['economic_risk']:,}
- City Health Score: {stats['city_health']}/100

TOP PRIORITY ROADS:
{queue_summary if queue_summary else '  No roads detected yet.'}

Answer the officer's question using this live data. Be specific with road names, numbers, and rupee amounts. Keep response concise (under 200 words)."""

    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            system=context,
            messages=[{"role": "user", "content": user_msg}]
        )
        reply = response.content[0].text
    except Exception as e:
        # Fallback rule-based reply
        reply = _rule_based_chat(user_msg, stats, queue)

    return jsonify({"reply": reply})

def _rule_based_chat(msg, stats, queue):
    msg = msg.lower()
    if "budget" in msg or "fix" in msg or "repair" in msg:
        if queue:
            top = queue[:3]
            ans = "Based on current data, I recommend fixing:\n"
            for i, r in enumerate(top, 1):
                ans += f"{i}. {r['road_name']} (Score {r['score']}/10) — ₹{r['economic_impact']:,} risk\n"
            return ans
    if "worst" in msg or "critical" in msg or "dangerous" in msg:
        if queue:
            r = queue[0]
            return (f"The most critical road is {r['road_name']} with score {r['score']}/10. "
                    f"Action required: {r['repair_action']}. "
                    f"Economic risk: ₹{r['economic_impact']:,}/30 days.")
    if "score" in msg or "health" in msg or "city" in msg:
        return (f"Current City Health Score is {stats['city_health']}/100. "
                f"There are {stats['critical']} critical roads and {stats['emergency']} emergency roads. "
                f"Total economic risk over 30 days: ₹{stats['economic_risk']:,}.")
    if "total" in msg or "how many" in msg:
        return (f"I have detected {stats['total']} road defects so far. "
                f"{stats['critical']} are critical (score 6+) and "
                f"{stats['emergency']} are emergency (score 8+).")
    return ("I can help you with repair recommendations, budget planning, and road priority analysis. "
            "Try asking: 'Which road should I fix first?' or 'What is the city health score?'")

# ─── Add Sample Data (for Demo) ───────────────────────────────────────────────
@app.route("/api/add-demo-data", methods=["POST"])
@login_required
def api_add_demo():
    demo_records = [
        {"road": "MG Road",       "lat": 17.6868, "lng": 75.9079, "type": "pothole",            "conf": 0.91},
        {"road": "Station Road",  "lat": 17.6820, "lng": 75.9010, "type": "alligator_crack",    "conf": 0.84},
        {"road": "College Road",  "lat": 17.6900, "lng": 75.9150, "type": "transverse_crack",   "conf": 0.78},
        {"road": "Market Road",   "lat": 17.6780, "lng": 75.9000, "type": "longitudinal_crack", "conf": 0.70},
        {"road": "Ring Road",     "lat": 17.6950, "lng": 75.9200, "type": "pothole",            "conf": 0.95},
        {"road": "Nehru Nagar",   "lat": 17.6750, "lng": 75.9050, "type": "alligator_crack",    "conf": 0.88},
        {"road": "Gandhi Chowk",  "lat": 17.6810, "lng": 75.9120, "type": "pothole",            "conf": 0.93},
        {"road": "Tilak Road",    "lat": 17.6840, "lng": 75.9080, "type": "transverse_crack",   "conf": 0.75},
    ]
    for d in demo_records:
        cfg  = DAMAGE_CONFIG.get(d["type"], {"base": 5, "repair_cost": 10000, "label": d["type"]})
        size_mult = 1.3
        conf_w    = 1.0 if d["conf"] >= 0.9 else (0.85 if d["conf"] >= 0.7 else 0.7)
        score     = min(10.0, cfg["base"] * size_mult * conf_w)
        det       = [{
            "class": d["type"], "confidence": d["conf"],
            "bbox":  [50, 50, 200, 200],
            "label": cfg["label"], "single_score": round(score, 2),
            "repair_cost": cfg["repair_cost"],
        }]
        save_record(d["road"], d["lat"], d["lng"], det, round(score, 2), None, "demo")
    return jsonify({"ok": True, "added": len(demo_records), "total": len(road_records_db)})

# ─── Delete All Data (Reset) ──────────────────────────────────────────────────
@app.route("/api/reset", methods=["POST"])
@login_required
def api_reset():
    defects_db.clear()
    road_records_db.clear()
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTIVE INTELLIGENCE — DIFFERENTIATING FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Helper: Days since inspection ────────────────────────────────────────────
def _days_since(ts_str):
    try:
        ts = datetime.fromisoformat(ts_str)
        return (datetime.now() - ts).days
    except Exception:
        return 0

# ─── Helper: Monsoon vulnerability score ──────────────────────────────────────
def _monsoon_risk(record):
    """
    Monsoon Risk = weighted combination of:
    - Current severity (40%)
    - Crack types present (transverse/alligator are water entry points) (30%)
    - Days since last inspection (older = unknown state = risky) (20%)
    - Score trend proxy (10%)
    Returns 0-100.
    """
    score   = record.get("score", 0)
    days    = _days_since(record.get("timestamp", datetime.now().isoformat()))
    types   = [d.get("label","") for d in record.get("all_detections", record.get("detections", []))]

    # Crack type vulnerability weight
    crack_w = 0
    for t in types:
        tl = t.lower()
        if "transverse" in tl:  crack_w = max(crack_w, 0.9)
        elif "alligator" in tl: crack_w = max(crack_w, 0.85)
        elif "longitudinal" in tl: crack_w = max(crack_w, 0.5)
        elif "pothole" in tl:   crack_w = max(crack_w, 0.7)

    severity_factor  = (score / 10) * 40
    crack_factor     = crack_w * 30
    age_factor       = min(20, (days / 90) * 20)
    base_factor      = 10

    return round(min(100, severity_factor + crack_factor + age_factor + base_factor), 1)

# ─── Helper: Deterioration prediction ─────────────────────────────────────────
def _predict_deterioration(record):
    """
    Estimates what the road score will be in 30 / 60 / 90 days
    using a simple decay model:
    - Potholes: fast growth (score × 1.3 per 30 days)
    - Alligator: medium growth (× 1.2)
    - Cracks: slow growth (× 1.1)
    - Good roads: slow natural wear (× 1.05)
    """
    current = record.get("score", 0)
    types   = [d.get("label","") for d in record.get("all_detections", record.get("detections",[]))]
    type_str = " ".join(types).lower()

    if "pothole" in type_str:          rate = 1.30
    elif "alligator" in type_str:      rate = 1.22
    elif "transverse" in type_str:     rate = 1.15
    elif "longitudinal" in type_str:   rate = 1.10
    else:                              rate = 1.05

    p30 = round(min(10.0, current * rate), 2)
    p60 = round(min(10.0, current * rate**2), 2)
    p90 = round(min(10.0, current * rate**3), 2)

    # Predicted level at 90 days
    def _lv(s):
        if s >= 8: return "Emergency"
        if s >= 6: return "Critical"
        if s >= 4: return "Severe"
        if s >= 2: return "Moderate"
        return "Good"

    # Days until it hits Emergency (score 8) if untreated
    days_to_emergency = None
    if current < 8 and rate > 1.0:
        import math
        try:
            days_to_emergency = round((math.log(8 / current) / math.log(rate)) * 30)
        except Exception:
            pass

    return {
        "current": current,
        "in_30_days": p30, "level_30": _lv(p30),
        "in_60_days": p60, "level_60": _lv(p60),
        "in_90_days": p90, "level_90": _lv(p90),
        "days_to_emergency": days_to_emergency,
        "deterioration_rate": rate,
    }

# ─── API: Predictive Analysis ─────────────────────────────────────────────────
@app.route("/api/predictive-analysis")
@login_required
def api_predictive():
    """
    For each road in DB:
    - Predict score at 30/60/90 days
    - Calculate monsoon risk
    - Flag which currently-safe roads will become critical before next monsoon
    - Rank by urgency (not current score, but predicted future damage)
    """
    results = []
    # Group by road, take latest record per road
    roads = {}
    for r in road_records_db:
        rn = r["road_name"]
        if rn not in roads or r["timestamp"] > roads[rn]["timestamp"]:
            roads[rn] = r

    for rn, r in roads.items():
        pred   = _predict_deterioration(r)
        m_risk = _monsoon_risk(r)
        days   = _days_since(r.get("timestamp", datetime.now().isoformat()))

        # "Hidden danger" = roads that look OK now but will be critical soon
        currently_safe  = r["score"] < 6
        will_be_critical = pred["in_60_days"] >= 6
        hidden_danger   = currently_safe and will_be_critical

        results.append({
            "id":             r["id"],
            "road_name":      rn,
            "current_score":  r["score"],
            "current_level":  r["level"],
            "color":          r["color"],
            "prediction":     pred,
            "monsoon_risk":   m_risk,
            "days_since_inspection": days,
            "hidden_danger":  hidden_danger,
            "economic_impact_now":    r["economic_impact"],
            "economic_impact_90d":    calc_economic_impact(pred["in_90_days"]),
            "inspection_date": r.get("inspection_date",""),
            "source":          r.get("source",""),
            "annotated_img":   r.get("annotated_img") or r.get("worst_img"),
        })

    # Sort: hidden dangers first, then by monsoon risk desc
    results.sort(key=lambda x: (not x["hidden_danger"], -x["monsoon_risk"]))
    return jsonify(results)

# ─── API: Monsoon Report ──────────────────────────────────────────────────────
@app.route("/api/monsoon-report")
@login_required
def api_monsoon_report():
    """
    Generates a city-wide monsoon preparedness report:
    - Overall preparedness score
    - Roads at extreme risk during monsoon
    - Estimated cost to make city monsoon-safe
    - Roads that MUST be fixed before monsoon
    """
    roads = {}
    for r in road_records_db:
        rn = r["road_name"]
        if rn not in roads or r["timestamp"] > roads[rn]["timestamp"]:
            roads[rn] = r

    risk_list = []
    total_cost = 0
    for rn, r in roads.items():
        m_risk = _monsoon_risk(r)
        pred   = _predict_deterioration(r)
        # Repair cost estimate (rough: ₹50k per severity point × road)
        repair_cost = max(50000, int(r["score"] * 50000))
        if m_risk >= 50:
            total_cost += repair_cost
        risk_list.append({
            "road_name":      rn,
            "monsoon_risk":   m_risk,
            "current_score":  r["score"],
            "predicted_score_monsoon": pred["in_60_days"],  # 2 months = monsoon peak
            "color":          r["color"],
            "repair_cost":    repair_cost,
            "must_fix":       m_risk >= 70,
            "watch":          50 <= m_risk < 70,
            "safe":           m_risk < 50,
        })

    risk_list.sort(key=lambda x: -x["monsoon_risk"])

    must_fix   = [r for r in risk_list if r["must_fix"]]
    watch_list = [r for r in risk_list if r["watch"]]
    safe_list  = [r for r in risk_list if r["safe"]]

    overall_prep = 0
    if risk_list:
        avg_risk = sum(r["monsoon_risk"] for r in risk_list) / len(risk_list)
        overall_prep = round(100 - avg_risk, 1)

    return jsonify({
        "overall_preparedness": overall_prep,
        "total_roads":    len(risk_list),
        "must_fix_count": len(must_fix),
        "watch_count":    len(watch_list),
        "safe_count":     len(safe_list),
        "estimated_repair_cost": total_cost,
        "must_fix":       must_fix,
        "watch_list":     watch_list,
        "safe_list":      safe_list,
        "generated_at":   datetime.now().strftime("%d %b %Y %H:%M"),
    })

# ─── API: Zone Risk Intelligence ─────────────────────────────────────────────
@app.route("/api/zone-intelligence")
@login_required
def api_zone_intelligence():
    """
    Groups all roads by geographic zone (0.01 degree grid ≈ 1km zones).
    Returns zone-level risk summary — politicians & planners think in zones.
    """
    zones = {}
    for r in road_records_db:
        if not r.get("lat") or not r.get("lng"):
            continue
        # 0.01 degree grid key
        zone_lat = round(r["lat"], 2)
        zone_lng = round(r["lng"], 2)
        key = f"{zone_lat}_{zone_lng}"
        if key not in zones:
            zones[key] = {
                "zone_id":    key,
                "center_lat": zone_lat,
                "center_lng": zone_lng,
                "roads":      [],
                "zone_name":  f"Zone {len(zones)+1}",
            }
        zones[key]["roads"].append(r)

    result = []
    for key, z in zones.items():
        roads   = z["roads"]
        n       = len(roads)
        avg_sc  = round(sum(r["score"] for r in roads) / n, 2) if n else 0
        worst   = max(roads, key=lambda x: x["score"]) if roads else {}
        total_e = sum(r["economic_impact"] for r in roads)
        crit_n  = sum(1 for r in roads if r["score"] >= 6)
        m_risks = [_monsoon_risk(r) for r in roads]
        avg_m   = round(sum(m_risks)/len(m_risks), 1) if m_risks else 0

        level, bs, color = get_level(avg_sc)

        result.append({
            "zone_id":        key,
            "zone_name":      z["zone_name"],
            "center_lat":     z["center_lat"],
            "center_lng":     z["center_lng"],
            "road_count":     n,
            "avg_score":      avg_sc,
            "level":          level,
            "color":          color,
            "critical_roads": crit_n,
            "total_economic_risk": total_e,
            "avg_monsoon_risk": avg_m,
            "worst_road":     worst.get("road_name",""),
            "worst_score":    worst.get("score", 0),
        })

    result.sort(key=lambda x: -x["avg_score"])
    return jsonify(result)

# ─── Page routes for new features ─────────────────────────────────────────────
@app.route("/predictive")
@login_required
def predictive_page():
    return render_template("predictive.html", officer=session.get("name"))

@app.route("/monsoon")
@login_required
def monsoon_page():
    return render_template("monsoon.html", officer=session.get("name"))

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RoadSense AI — Starting Server")
    print("  Dashboard : http://localhost:5000/dashboard")
    print("  City Map  : http://localhost:5000/city-map")
    print("  Citizen   : http://localhost:5000/report")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)