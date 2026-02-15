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

# analyse class distribution
train_lbl_dir = os.path.join(DATASET_ROOT, "train", "labels")
class_counts = {0:0, 1:0} # 0= dents, 1=scratches
for label_file in os.listdir(train_lbl_dir):
  if not label_file.endswith(".txt"):
    continue

  with open(os.path.join(train_lbl_dir, label_file), 'r') as f:
    for line in f:
      class_id = int(line.split()[0])
      class_counts[class_id] += 1
total_instances = sum(class_counts.values())
dent_ratio = class_counts[0] / total_instances
scratch_ratio = class_counts[1] / total_instances
print(f"Dents: {class_counts[0]:4} instances ({dent_ratio*100:.1f}%)")
print(f"Scratches: {class_counts[1]:4} instances ({scratch_ratio*100:.1f}%)")
print(f"Imbalance ratio: {class_counts[1]/class_counts[0]:.2f}:1")

# shows class imbalance so proves copy_paste and mosaic settings
# model training
model = YOLO("yolov8m-seg.pt")
model.train(
    data=os.path.join(DATASET_ROOT, "data.yaml"),
    epochs=100,
    patience=20,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    cache="ram",
    amp=True,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    cos_lr=True,
    mosaic=0.9,
    close_mosaic=10,
    mixup=0.2,
    copy_paste=0.5,
    degrees=15,
    translate=0.15,
    scale=0.7,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    box=7.5,
    cls=2.5,
    dfl=1.5,
    plots=True,
    save=True,
    save_period=10,
    exist_ok=True,
    verbose=True,
)

# save model
model_path = os.path.join(MODEL_DIR, "v2.pt")
model.save(model_path)
print (f"Model saved at: {model_path}")

# validation
metrics = model.val()
print("Segmentation Metrics")
print(f"mAP50(mask): {metrics.seg.map50:.3f}")
print(f"mAP50-95: {metrics.seg.map:.3f}")
print(f"Precision(mask): {metrics.seg.mp:.3f}")
print(f"Recall(mask): {metrics.seg.mr:.3f}")

