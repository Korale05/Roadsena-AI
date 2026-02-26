# ai_engine.py — FIXED VERSION
from ultralytics import YOLO
import os

# ── Load model once when server starts ──────────────────────────────────────
model = YOLO(
    r"D:\Pathhole_Detection\YOLOv8_Pothole_Segmentation_Road_Damage_Assessment\model\best.pt"
)

# ── What the model can detect ────────────────────────────────────────────────
# This model detects "Pothole" — the class name depends on what it was trained on
# Print model.names to check: print(model.names)
DAMAGE_CLASSES = {
    'D00': 'Longitudinal Crack',
    'D10': 'Transverse Crack',
    'D20': 'Alligator Crack',
    'D40': 'Pothole',
    'pothole': 'Pothole',      # ← fallback for pothole-only models
    'crack':   'Crack',        # ← fallback
}

BASE_SCORE = {
    'D40': 9, 'D20': 7, 'D10': 5, 'D00': 3,
    'pothole': 9, 'crack': 4   # ← fallback scores
}


def analyze_image(image_path: str):
    """
    WHAT THIS FUNCTION DOES — Step by step:
    
    1. Takes a road image path as input
    2. Runs YOLOv8 model → finds all damage in the image
    3. For each damage found → records type, confidence, size
    4. Calculates severity score (0-10) using calculate_score()
    5. Saves a new image with colored boxes drawn on damage
    6. Returns everything as a dictionary
    
    INPUT:  "uploads/temp_abc123.jpg"
    OUTPUT: { detections: [...], score: {...}, annotated_image: "annotated_abc123.jpg" }
    """
    
    # ── STEP 1: Run the AI model on the image ───────────────────────────────
    # conf=0.25 means "only report detections where AI is 25%+ confident"
    results  = model(image_path, conf=0.25, verbose=False)
    
    # Get image dimensions (height × width in pixels)
    img_h, img_w = results[0].orig_shape[0], results[0].orig_shape[1]
    img_area = img_h * img_w   # total pixel area of image
    
    # ── STEP 2: Loop through every detected damage box ──────────────────────
    detections = []
    
    for box in results[0].boxes:
        # What class/type did the model detect?
        raw_class  = model.names[int(box.cls)]       # e.g. "pothole" or "D40"
        class_name = DAMAGE_CLASSES.get(raw_class, raw_class)  # human readable name
        
        # How confident is the AI? (0-100%)
        confidence = round(float(box.conf) * 100, 1)
        
        # Where is the damage? (bounding box coordinates)
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        
        # How big is the damage relative to the whole image?
        # Example: size_ratio=0.05 means damage covers 5% of the image
        defect_area = (x2 - x1) * (y2 - y1)
        size_ratio  = round(defect_area / img_area, 4)
        
        detections.append({
            "class_id":   raw_class,    # "pothole"
            "type":       class_name,   # "Pothole"
            "confidence": confidence,   # 87.3
            "bbox":       [x1, y1, x2, y2],  # [120, 45, 380, 290]
            "size_ratio": size_ratio    # 0.08 = 8% of image
        })
    
    # ── STEP 3: Save the annotated image (with boxes drawn on it) ───────────
    # Create a UNIQUE filename so images don't overwrite each other
    import uuid
    unique_name    = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
    os.makedirs("outputs", exist_ok=True)
    out_path       = os.path.join("outputs", unique_name)
    results[0].save(filename=out_path)   # YOLOv8 draws boxes and saves
    
    # ── STEP 4: Calculate severity score ────────────────────────────────────
    score_result = calculate_score(detections)
    
    # ── STEP 5: Return everything ────────────────────────────────────────────
    return {
        "detections":      detections,
        "score":           score_result,
        "annotated_image": unique_name   # ← JUST the filename, NOT full path
        #                                    main.py adds "outputs/" when serving
    }


def calculate_score(detections: list) -> dict:
    """
    WHAT THIS FUNCTION DOES — Explained simply:
    
    Think of it like grading a road's health from 0 to 10.
    
    Each damage gets a score based on 3 things:
      1. TYPE    — pothole (worst=9) vs crack (least=3)
      2. SIZE    — large damage scores higher than small
      3. CONFIDENCE — if AI is very sure (90%+), full score
                      if AI is less sure (50-70%), reduced score
    
    Then we average all scores and add a penalty for multiple defects.
    
    EXAMPLE:
      1 large pothole, AI 90% sure:
        base=9 × size=1.6 × conf=1.0 = 14.4 → capped at 10
        Final severity = 10 → Emergency
    
      2 small cracks, AI 75% sure:
        crack1: base=4 × size=1.0 × conf=0.85 = 3.4
        crack2: base=4 × size=1.0 × conf=0.85 = 3.4
        average = 3.4, count penalty = 1.1x
        Final severity = 3.7 → Moderate
    """
    
    # If nothing detected → road is perfect
    if not detections:
        return {
            "severity_score":     0,
            "road_quality_index": 100,
            "severity_level":     "Good",
            "color":              "green",
            "defect_count":       0,
            "dominant_damage":    "None",
            "repair_urgency":     "No action needed",
            "economic_impact_rs": 0,
            "breakdown":          []
        }
    
    scores    = []
    breakdown = []
    
    for d in detections:
        # 1. Base danger score of this damage type
        base = BASE_SCORE.get(d['class_id'], 5)
        
        # 2. Size multiplier — bigger damage is more dangerous
        if   d['size_ratio'] > 0.06: size = 1.6   # Large
        elif d['size_ratio'] > 0.02: size = 1.3   # Medium
        else:                        size = 1.0   # Small
        
        # 3. Confidence weight — how sure is the AI?
        if   d['confidence'] >= 90: conf = 1.00
        elif d['confidence'] >= 70: conf = 0.85
        elif d['confidence'] >= 50: conf = 0.70
        else:                       conf = 0.55
        
        # Score for this ONE detection (max 10)
        single = min(10, base * size * conf)
        scores.append(single)
        breakdown.append({
            "type":         d['type'],
            "single_score": round(single, 2),
            "confidence":   d['confidence'],
            "size":         "Large" if d['size_ratio']>0.06 else "Medium" if d['size_ratio']>0.02 else "Small"
        })
    
    # Multiple defects = worse road (10% penalty per extra, max 40%)
    count_penalty  = 1 + min(0.4, (len(detections) - 1) * 0.10)
    avg            = sum(scores) / len(scores)
    severity       = min(10.0, round(avg * count_penalty, 1))
    quality_index  = max(0, round(100 - severity * 10))
    
    # ── Severity label + color + urgency ────────────────────────────────────
    #   Score 0-2  → Good      → green
    #   Score 2-4  → Moderate  → yellow
    #   Score 4-6  → Severe    → orange
    #   Score 6-8  → Critical  → red
    #   Score 8-10 → Emergency → darkred
    if   severity <= 2: level, color, urgency = "Good",      "green",   "Monitor monthly"
    elif severity <= 4: level, color, urgency = "Moderate",  "yellow",  "Repair within 90 days"
    elif severity <= 6: level, color, urgency = "Severe",    "orange",  "Repair within 30 days"
    elif severity <= 8: level, color, urgency = "Critical",  "red",     "Repair within 7 days"
    else:               level, color, urgency = "Emergency", "darkred", "Repair IMMEDIATELY"
    
    # Which damage type is the most dangerous one found?
    dominant = max(detections, key=lambda x: BASE_SCORE.get(x['class_id'], 0))
    
    # Economic impact = severity × Rs.150/vehicle × 500 vehicles/day × 30 days
    # Example: severity 7 → 7 × 150 × 500 × 30 = Rs.15,750,000
    economic = round(severity * 150 * 500 * 30)
    
    return {
        "severity_score":     severity,        # 7.2
        "road_quality_index": quality_index,   # 28
        "severity_level":     level,           # "Critical"
        "color":              color,           # "red"
        "defect_count":       len(detections), # 3
        "dominant_damage":    dominant['type'],# "Pothole"
        "repair_urgency":     urgency,         # "Repair within 7 days"
        "economic_impact_rs": economic,        # 15750000
        "breakdown":          breakdown        # per-detection detail
    }