# main.py — how Member 2 (Backend) calls both files

from detect import detect_damage
from score  import calculate_score

def analyze_road_image(image_path):
    # Step 1: AI detects all damage
    detections, annotated_image = detect_damage(image_path)

    # Step 2: Score the damage
    result = calculate_score(detections)

    # Step 3: Return everything to frontend
    return {
        "annotated_image": annotated_image,
        "detections":      detections,
        "score":           result
    }

img_path = r"images.jpg"
res = analyze_road_image(image_path=img_path)
print("\n===== ROAD DAMAGE REPORT =====")
print(f"Annotated Image : {res['annotated_image']}")
print(f"Total Defects   : {res['score']['defect_count']}")
print(f"Severity Score  : {res['score']['severity_score']}")
print(f"Road Quality    : {res['score']['road_quality_index']}")
print(f"Severity Level  : {res['score']['severity_level']}")
print(f"Repair Urgency  : {res['score']['repair_urgency']}")
print(f"Economic Impact : ₹{res['score']['economic_impact']:,}")
print("================================\n")

print("---- Defect Breakdown ----")
for i, d in enumerate(res['score']['breakdown'], 1):
    print(f"{i}. {d['type']}")
    print(f"   Confidence  : {d['confidence']}%")
    print(f"   Single Score: {d['single_score']}")
    print()


# Example Output:
# {
#   "annotated_image": "output_road.jpg",
#   "detections": [
#       {"type": "Pothole",    "confidence": 91.3, "size_ratio": 0.07},
#       {"type": "Transverse Crack", "confidence": 78.5, "size_ratio": 0.02}
#   ],
#   "score": {
#       "severity_score":     8.2,
#       "road_quality_index": 18,
#       "severity_level":     "Critical",
#       "color":              "red",
#       "defect_count":       2,
#       "dominant_damage":    "Pothole",
#       "repair_urgency":     "Repair within 7 days",
#       "economic_impact":    2250000
#   }
# }
