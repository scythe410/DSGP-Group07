import os
from huggingface_hub import hf_hub_download, snapshot_download
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT_DIR, '.env'))

def bootstrap_models():
    """
    Downloads ML models from HuggingFace Hub if not already present locally.
    This runs PRE-STARTUP so it doesn't timeout Uvicorn's event loop.
    In HuggingFace Space deployments, model weights are gitignored and must
    be fetched at startup.
    """
    hf_token = os.environ.get("HF_TOKEN")  # Set as Space Secret in HF dashboard

    yolo_path  = os.path.join(ROOT_DIR, "damage-detection", "models", "v2.pt")
    seg_path   = os.path.join(ROOT_DIR, "damage-detection", "models", "best_model")
    xgb_path   = os.path.join(ROOT_DIR, "price-model", "best_optimized_model.pkl")
    prep_path  = os.path.join(ROOT_DIR, "price-model", "preprocessing_optimized.pkl")

    # YOLO v2 damage detection model
    if not os.path.exists(yolo_path):
        print(f"[BOOTSTRAP] Downloading YOLO v2 model from HF Hub to {yolo_path}...")
        os.makedirs(os.path.dirname(yolo_path), exist_ok=True)
        hf_hub_download(
            repo_id="scythe410/vehicle-damage-detection-yolo",
            filename="v2.pt",
            local_dir=os.path.dirname(yolo_path),
            token=hf_token,
        )
        print("[BOOTSTRAP] YOLO model ready.")

    # SegFormer semantic segmentation model
    seg_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
    if not os.path.exists(seg_path) or not all(
        os.path.exists(os.path.join(seg_path, f)) for f in seg_files
    ):
        print(f"[BOOTSTRAP] Downloading SegFormer model from HF Hub to {seg_path}...")
        snapshot_download(
            repo_id="scythe410/vehicle-damage-detection-sagformer",
            local_dir=seg_path,
            token=hf_token,
            ignore_patterns=["*.md", ".gitattributes"],
        )
        print("[BOOTSTRAP] SegFormer model ready.")

    # XGBoost price prediction model + preprocessing pipeline
    if not os.path.exists(xgb_path):
        print("[BOOTSTRAP] Downloading XGBoost price model from HF Hub...")
        os.makedirs(os.path.dirname(xgb_path), exist_ok=True)
        hf_hub_download(
            repo_id="scythe410/sri-lankan-vehicle-price-prediction",
            filename="best_optimized_model.pkl",
            local_dir=os.path.dirname(xgb_path),
            token=hf_token,
        )
        hf_hub_download(
            repo_id="scythe410/sri-lankan-vehicle-price-prediction",
            filename="preprocessing_optimized.pkl",
            local_dir=os.path.dirname(xgb_path),
            token=hf_token,
        )
        print("[BOOTSTRAP] XGBoost models ready.")

    # Anomaly detection model (OneClassSVM) — guards both pipeline data quality
    # and user input validation in the /predict_price endpoint
    anomaly_path = os.path.join(ROOT_DIR, "price-model", "anomaly_model.pkl")
    if not os.path.exists(anomaly_path):
        print("[BOOTSTRAP] Downloading anomaly detection model from HF Hub...")
        os.makedirs(os.path.dirname(anomaly_path), exist_ok=True)
        hf_hub_download(
            repo_id="scythe410/sri-lankan-vehicle-anomaly-detection",
            filename="anomaly_model.pkl",
            local_dir=os.path.dirname(anomaly_path),
            token=hf_token,
        )
        print("[BOOTSTRAP] Anomaly model ready.")

if __name__ == "__main__":
    bootstrap_models()
