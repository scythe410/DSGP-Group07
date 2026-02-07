# NOTE:
""" This script is intended to be run on a GPU machine (e.g. Google Colab with T4 GPU)
 Local CPU execution is only for testing and will be extremely slow  """


import os
from ultralytics import YOLO
from roboflow import Roboflow

DATASET_ROOT = "Car-Damage-Analizer-1"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Roboflow authentication
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if ROBOFLOW_API_KEY is None:
    raise ValueError("RoboFlow API key not set")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("car-4ghb2").project("car-damage-analizer")
version = project.version(1)
dataset = version.download("yolov8")

# checking dataset
splits = ["train", "valid", "test"]
for split in splits:
    img_dir = os.path.join(DATASET_ROOT, split,"images")
    lbl_dir = os.path.join(DATASET_ROOT, split,"labels")
    n_imgs = len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    n_lbls = len([f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")])
    print(f"{split.upper():6} | Images: {n_imgs:4} | Labels: {n_lbls:4} | Match: {n_imgs == n_lbls}")

# model training
model = YOLO("yolov8m-seg.pt")
model.train(
    data = os.path.join(DATASET_ROOT, "data.yaml"),
    epochs = 80,
    imgsz = 640,
    batch = 16,
    device = 0,
    workers = 2,
    optimizer = "SGD",
    lr0 = 0.01,
    cos_lr = True,
    amp = True,
    patience = 10,
    mosaic = 0.0,
    close_mosaic = 0
)

# save model
model_path = os.path.join(MODEL_DIR, "v1.pt")
model.save(model_path)
print (f"Model saved at: {model_path}")

# validation
metrics = model.val()
print("Segmentation Metrics")
print(f"mAP50(mask): {metrics.seg.map50:.3f}")
print(f"mAP50-95: {metrics.seg.map:.3f}")
print(f"Precision(mask): {metrics.seg.mp:.3f}")
print(f"Recall(mask): {metrics.seg.mr:.3f}")

