# detect.py — Member 1 writes this file
from ultralytics import YOLO
import cv2
import os

# Load model ONCE when app starts (saves time)
model = YOLO(
    r"D:\Pathhole_Detection\YOLOv8_Pothole_Segmentation_Road_Damage_Assessment\model\best.pt"
)


# The 4 damage classes this model knows
DAMAGE_CLASSES = {
    'D00': 'Longitudinal Crack',
    'D10': 'Transverse Crack', 
    'D20': 'Alligator Crack',
    'D40': 'Pothole'
}

def detect_damage(image_path):
    """
    Input:  any road image path
    Output: list of damage found + annotated image saved
    """
    results = model(image_path, conf=0.25)
    
    image_area = results[0].orig_shape[0] * results[0].orig_shape[1]
    detections = []

    for box in results[0].boxes:
        class_id   = model.names[int(box.cls)]   # e.g. "D40"
        class_name = DAMAGE_CLASSES.get(class_id, class_id)
        confidence = round(float(box.conf) * 100, 1)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Size of damage relative to image (0.0 to 1.0)
        defect_area = (x2 - x1) * (y2 - y1)
        size_ratio  = round(defect_area / image_area, 4)

        detections.append({
            "class_id":   class_id,       # "D40"
            "type":       class_name,     # "Pothole"
            "confidence": confidence,     # 91.3
            "bbox":       [x1, y1, x2, y2],
            "size_ratio": size_ratio      # 0.08 means 8% of image
        })

    # Save the annotated image (boxes drawn on it)
    output_path = "output_" + os.path.basename(image_path)
    results[0].save(filename=output_path)

    return detections, output_path

det = []

img_path = r"upload\path2.jpg"
det , path = detect_damage(image_path=img_path)