# score.py — Member 1 writes this file
# PURPOSE: Take detections from detect.py → output a severity score

# ─── HOW SCORING WORKS (Read This First) ──────────────────────────────
#
# Each damage type has a BASE DANGER LEVEL (1-10):
#   Pothole          = 9  (most dangerous, damages vehicles immediately)
#   Alligator Crack  = 7  (will become pothole soon)
#   Transverse Crack = 5  (moderate — water gets in and spreads)
#   Longitudinal Crack = 3 (least urgent — structural but slow)
#
# Each detection also gets a SIZE MULTIPLIER:
#   Small (< 2% of image)  = 1.0x
#   Medium (2–6% of image) = 1.3x
#   Large  (> 6% of image) = 1.6x
#
# Each detection also has CONFIDENCE WEIGHT:
#   AI 90%+ sure = full weight (1.0)
#   AI 70–90%    = 0.85 weight
#   AI < 70%     = 0.7 weight
#
# FINAL SCORE = average of all (base × size × confidence) capped at 10
# ROAD QUALITY INDEX = 100 - (score × 10)  →  higher is better
# ──────────────────────────────────────────────────────────────────────

# Base danger level per damage type
BASE_SCORE = {
    'D40': 9,   # Pothole
    'D20': 7,   # Alligator Crack
    'D10': 5,   # Transverse Crack
    'D00': 3,   # Longitudinal Crack
}

def get_size_multiplier(size_ratio):
    """How big is the damage compared to full image?"""
    if size_ratio > 0.06:    # bigger than 6% of image = LARGE
        return 1.6
    elif size_ratio > 0.02:  # 2–6% = MEDIUM
        return 1.3
    else:                    # under 2% = SMALL
        return 1.0

def get_confidence_weight(confidence):
    """How sure is the AI about this detection?"""
    if confidence >= 90:
        return 1.0
    elif confidence >= 70:
        return 0.85
    else:
        return 0.7

def calculate_score(detections):
    """
    Input:  list of detections from detect.py
    Output: full scoring result dictionary
    """
    # If nothing detected → road is perfect
    if not detections:
        return {
            "severity_score":    0,
            "road_quality_index": 100,
            "severity_level":    "Good",
            "color":             "green",
            "defect_count":      0,
            "dominant_damage":   "None",
            "repair_urgency":    "No repair needed",
            "economic_impact":   0,
            "breakdown":         []
        }

    scores_per_detection = []
    breakdown = []

    for d in detections:
        base       = BASE_SCORE.get(d['class_id'], 5)
        size_mult  = get_size_multiplier(d['size_ratio'])
        conf_weight= get_confidence_weight(d['confidence'])

        # Score for THIS ONE detection
        single_score = min(10, base * size_mult * conf_weight)
        scores_per_detection.append(single_score)

        breakdown.append({
            "type":         d['type'],
            "single_score": round(single_score, 2),
            "base":         base,
            "size_mult":    size_mult,
            "confidence":   d['confidence']
        })

    # Multiple defects = worse road (add 10% per extra defect, max 40%)
    count_penalty = 1 + min(0.4, (len(detections) - 1) * 0.10)

    # Final severity = average score × count penalty, capped at 10
    avg_score      = sum(scores_per_detection) / len(scores_per_detection)
    severity_score = min(10, round(avg_score * count_penalty, 1))

    # Road Quality Index (0–100, higher = better road)
    road_quality_index = max(0, round(100 - (severity_score * 10)))

    # Severity level label + color for UI
    if severity_score <= 2:
        level, color, urgency = "Minor",    "green",  "Monitor monthly"
    elif severity_score <= 4:
        level, color, urgency = "Moderate", "yellow", "Repair within 90 days"
    elif severity_score <= 6:
        level, color, urgency = "Severe",   "orange", "Repair within 30 days"
    elif severity_score <= 8:
        level, color, urgency = "Critical", "red",    "Repair within 7 days"
    else:
        level, color, urgency = "Emergency","darkred","Repair IMMEDIATELY"

    # Find the most dangerous damage type found
    dominant = max(detections, key=lambda x: BASE_SCORE.get(x['class_id'], 0))

    # Economic impact in Rupees
    # Logic: higher severity = more vehicle damage per day × estimated traffic
    daily_damage_per_vehicle = severity_score * 150  # ₹150–₹1350 per vehicle
    estimated_vehicles_per_day = 500                 # assume medium traffic road
    days_until_repaired = 30                         # assume 30-day delay
    economic_impact = round(
        daily_damage_per_vehicle * estimated_vehicles_per_day * days_until_repaired
    )

    return {
        "severity_score":     severity_score,          # 0–10
        "road_quality_index": road_quality_index,      # 0–100
        "severity_level":     level,                   # "Critical"
        "color":              color,                   # "red"
        "defect_count":       len(detections),         # 3
        "dominant_damage":    dominant['type'],        # "Pothole"
        "repair_urgency":     urgency,                 # "Repair within 7 days"
        "economic_impact":    economic_impact,         # 2250000 (in ₹)
        "breakdown":          breakdown                # per-detection detail
    }