import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

print("Loading model from:", MODEL_PATH)

model = YOLO(MODEL_PATH)