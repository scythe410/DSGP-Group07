"""
One-time upload: pushes anomaly_model.pkl to HuggingFace Hub.
"""
import os, sys
from huggingface_hub import HfApi

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH  = os.path.join(ROOT_DIR, "price-model", "anomaly_model.pkl")
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
REPO_ID   = "scythe410/sri-lankan-vehicle-anomaly-detection"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set."); sys.exit(1)
if not os.path.exists(PKL_PATH):
    print(f"ERROR: {PKL_PATH} not found. Run fit_anomaly_model.py first."); sys.exit(1)

api = HfApi(token=HF_TOKEN)
print(f"Uploading {PKL_PATH} → {REPO_ID} ...")
api.upload_file(
    path_or_fileobj=PKL_PATH,
    path_in_repo="anomaly_model.pkl",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add OneClassSVM anomaly detection model (fitted on 11,441 baseline listings)",
)
print("[OK] Upload complete.")
