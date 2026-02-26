# main.py — RoadSense AI Backend (Flask)
# Run with: python main.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, uuid, shutil
from datetime import datetime
from ai_engine import analyze_image

# ── SETUP ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow React frontend to connect

os.makedirs("outputs", exist_ok=True)   # create outputs folder if missing
os.makedirs("uploads", exist_ok=True)   # create uploads folder if missing

# In-memory database (stores all detections while server is running)
defects_db = []


# ── ROUTE 1: Home ──────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "RoadSense AI API is running!",
        "version": "1.0",
        "endpoints": {
            "upload_image":     "POST /api/upload-image",
            "get_defects":      "GET  /api/defects",
            "heatmap_data":     "GET  /api/heatmap-data",
            "city_score":       "GET  /api/city-health-score",
            "priority_queue":   "GET  /api/priority-queue",
            "get_image":        "GET  /outputs/<filename>"
        }
    })

# Add this route to main.py
@app.route("/inspect")
def inspect_page():
    from flask import send_from_directory
    return send_from_directory(".", "index.html")



# ── ROUTE 2: Upload Image → Run AI → Return Results ────────────────────────
@app.route("/api/upload-image", methods=["POST"])
def upload_image():
    # ── Step 1: Check if image was sent ──
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No image file sent"}), 400

    file      = request.files["file"]
    lat       = float(request.form.get("lat", 0.0))
    lng       = float(request.form.get("lng", 0.0))
    road_name = request.form.get("road_name", "Unknown Road")

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    # ── Step 2: Save the uploaded image temporarily ──
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    temp_path     = os.path.join("uploads", temp_filename)
    file.save(temp_path)

    # ── Step 3: Run AI analysis ──
    try:
        result = analyze_image(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"success": False, "error": str(e)}), 500

    # ── Step 4: Clean up temp file ──
    os.remove(temp_path)

    # ── Step 5: Build the record and save to in-memory database ──
    record = {
        "id":              str(uuid.uuid4()),
        "road_name":       road_name,
        "lat":             lat,
        "lng":             lng,
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detections":      result["detections"],
        "score":           result["score"],
        "annotated_image": result["annotated_image"],
        "status":          "pending_repair"
    }
    defects_db.append(record)

    # ── Step 6: Print to terminal so you can see it working ──
    print("\n" + "="*50)
    print(f"NEW DETECTION — {road_name}")
    print(f"Defects Found : {result['score']['defect_count']}")
    print(f"Severity Score: {result['score']['severity_score']}/10")
    print(f"Road Quality  : {result['score']['road_quality_index']}/100")
    print(f"Status        : {result['score']['severity_level']}")
    print(f"Repair Urgency: {result['score']['repair_urgency']}")
    print(f"Economic Impact: Rs.{result['score']['economic_impact_rs']:,}")
    print("="*50 + "\n")

    return jsonify({"success": True, "data": record})


# ── ROUTE 3: Serve Annotated Images ────────────────────────────────────────
@app.route("/outputs/<filename>", methods=["GET"])
def get_image(filename):
    from flask import send_from_directory
    return send_from_directory("outputs", filename)


# ── ROUTE 4: Get All Defects ───────────────────────────────────────────────
@app.route("/api/defects", methods=["GET"])
def get_defects():
    return jsonify({
        "total": len(defects_db),
        "defects": defects_db
    })


# ── ROUTE 5: Heatmap Data (GPS points for the map) ────────────────────────
@app.route("/api/heatmap-data", methods=["GET"])
def get_heatmap():
    points = []
    for d in defects_db:
        points.append({
            "lat":       d["lat"],
            "lng":       d["lng"],
            "intensity": d["score"]["severity_score"],
            "level":     d["score"]["severity_level"],
            "color":     d["score"]["color"],
            "road":      d["road_name"],
            "timestamp": d["timestamp"],
            "id":        d["id"]
        })
    return jsonify({"total_points": len(points), "points": points})


# ── ROUTE 6: City Health Score ─────────────────────────────────────────────
@app.route("/api/city-health-score", methods=["GET"])
def city_health():
    if not defects_db:
        return jsonify({
            "city_score":    100,
            "total_defects": 0,
            "total_roads":   0,
            "critical_count": 0,
            "alerts_sent":   0,
            "status":        "No data yet — upload road images to begin"
        })

    avg_quality  = sum(d["score"]["road_quality_index"] for d in defects_db) / len(defects_db)
    critical     = sum(1 for d in defects_db if d["score"]["severity_score"] >= 7)
    alerts_sent  = sum(1 for d in defects_db if d["score"]["severity_score"] >= 7)

    return jsonify({
        "city_score":    round(avg_quality),
        "total_defects": len(defects_db),
        "total_roads":   len(defects_db),
        "critical_count": critical,
        "alerts_sent":   alerts_sent,
        "status":        get_city_status(round(avg_quality))
    })

def get_city_status(score):
    if score >= 80: return "Good"
    if score >= 60: return "Moderate"
    if score >= 40: return "Severe"
    if score >= 20: return "Critical"
    return "Emergency"


# ── ROUTE 7: Priority Repair Queue ────────────────────────────────────────
@app.route("/api/priority-queue", methods=["GET"])
def priority_queue():
    # Sort by severity — worst road first
    sorted_defects = sorted(
        defects_db,
        key=lambda x: x["score"]["severity_score"],
        reverse=True
    )

    roads = []
    for i, d in enumerate(sorted_defects):
        roads.append({
            "rank":            i + 1,
            "road_name":       d["road_name"],
            "severity_score":  d["score"]["severity_score"],
            "severity_level":  d["score"]["severity_level"],
            "road_quality":    d["score"]["road_quality_index"],
            "defect_count":    d["score"]["defect_count"],
            "repair_urgency":  d["score"]["repair_urgency"],
            "economic_impact": d["score"]["economic_impact_rs"],
            "color":           d["score"]["color"],
            "timestamp":       d["timestamp"],
            "lat":             d["lat"],
            "lng":             d["lng"]
        })

    return jsonify({
        "total_roads": len(roads),
        "roads": roads
    })


# ── ROUTE 8: Get Single Defect by ID ──────────────────────────────────────
@app.route("/api/defects/<defect_id>", methods=["GET"])
def get_defect(defect_id):
    for d in defects_db:
        if d["id"] == defect_id:
            return jsonify(d)
    return jsonify({"error": "Defect not found"}), 404


# ── ROUTE 9: Statistics Summary ───────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def get_stats():
    if not defects_db:
        return jsonify({"message": "No data yet"})

    total_economic = sum(d["score"]["economic_impact_rs"] for d in defects_db)
    avg_severity   = sum(d["score"]["severity_score"] for d in defects_db) / len(defects_db)

    # Count by severity level
    level_counts = {"Good": 0, "Moderate": 0, "Severe": 0, "Critical": 0, "Emergency": 0}
    for d in defects_db:
        level = d["score"]["severity_level"]
        if level in level_counts:
            level_counts[level] += 1

    return jsonify({
        "total_inspections":   len(defects_db),
        "average_severity":    round(avg_severity, 1),
        "total_economic_risk": total_economic,
        "severity_breakdown":  level_counts,
        "most_common_damage":  get_most_common_damage()
    })

def get_most_common_damage():
    all_detections = []
    for d in defects_db:
        for det in d["detections"]:
            all_detections.append(det["type"])
    if not all_detections:
        return "None"
    return max(set(all_detections), key=all_detections.count)


# ── START SERVER ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  RoadSense AI — Backend Server")
    print("  Running at: http://localhost:5000")
    print("  Test APIs:  http://localhost:5000/api/defects")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)