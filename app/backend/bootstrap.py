import os
from huggingface_hub import hf_hub_download, snapshot_download
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT_DIR, '.env'))

def _is_real_file(path: str, min_bytes: int = 10_000) -> bool:
    """Returns True only if the path exists AND is larger than min_bytes.
    Git-LFS pointer files are ~130 bytes — this filters them out so we
    always download the real model weights instead of caching a pointer."""
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes

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
    if not _is_real_file(yolo_path, min_bytes=1_000_000):  # YOLO weights > 1 MB
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
        _is_real_file(os.path.join(seg_path, f)) for f in seg_files
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
    # Uses _is_real_file() to detect and re-download git-LFS pointer stubs
    # that HF Space may have copied into the container instead of real weights.
    if not _is_real_file(xgb_path) or not _is_real_file(prep_path):
        print("[BOOTSTRAP] Downloading XGBoost price model from HF Hub...")
        os.makedirs(os.path.dirname(xgb_path), exist_ok=True)
        for fname in ["best_optimized_model.pkl", "preprocessing_optimized.pkl"]:
            local = os.path.join(os.path.dirname(xgb_path), fname)
            if os.path.exists(local):
                os.remove(local)   # delete pointer stub before overwriting
            hf_hub_download(
                repo_id="scythe410/sri-lankan-vehicle-price-prediction",
                filename=fname,
                local_dir=os.path.dirname(xgb_path),
                token=hf_token,
            )
        print("[BOOTSTRAP] XGBoost models ready.")

    # Anomaly detection model (OneClassSVM)
    anomaly_path = os.path.join(ROOT_DIR, "price-model", "anomaly_model.pkl")
    if not _is_real_file(anomaly_path):
        print("[BOOTSTRAP] Downloading anomaly detection model from HF Hub...")
        os.makedirs(os.path.dirname(anomaly_path), exist_ok=True)
        if os.path.exists(anomaly_path):
            os.remove(anomaly_path)
        hf_hub_download(
            repo_id="scythe410/sri-lankan-vehicle-anomaly-detection",
            filename="anomaly_model.pkl",
            local_dir=os.path.dirname(anomaly_path),
            token=hf_token,
        )
        print("[BOOTSTRAP] Anomaly model ready.")


if __name__ == "__main__":
    bootstrap_models()
