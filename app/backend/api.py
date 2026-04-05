from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import re
import io
import pickle
import base64

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import pandas as pd
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Gemini model — lazily instantiated on first use so a bad model name
# doesn't crash the entire server at startup
_GEMINI_MODEL_NAME = 'gemini-2.5-flash'
_gemini_model = None

def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        _gemini_model = genai.GenerativeModel(_GEMINI_MODEL_NAME)
    return _gemini_model

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'price-model'))
sys.path.append(os.path.join(ROOT_DIR, 'data', 'pipeline'))

from predictor import predict_price

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Repair cost tiers based on SegFormer pixel area AS A PERCENTAGE of total image area.
# Using percentage makes tiers resolution-independent (works for phone photos, webcam, etc.)
# Each entry: (max_damage_pct, repair_action, cost_lkr). None = catch-all.
REPAIR_TIERS = [
    (1.5,  "Paintless Dent Repair",        5_000),   # <1.5%  — small ding/hail dent
    (6.0,  "Panel Beating",               12_000),   # <6.0%  — moderate dent, 1 panel
    (None, "Panel Replacement + Repaint", 35_000),   # ≥6.0%  — severe / multi-panel
]

# Minimum SegFormer pixel area to count as real surface damage (filters out noise).
# YOLO misses fine scratches — SegFormer is the fallback signal for paint/surface damage.
MIN_SEGFORMER_DAMAGE_PX = 500

GATEKEEPER_PROMPT = (
    "You are a strict vehicle image classifier. "
    "Your ONLY job is to determine if an image is a legitimate photograph of a motor vehicle "
    "or a vehicle part (e.g. car, truck, van, motorcycle, bumper, door panel, tyre). "
    "If the image shows anything else — including animals, people, food, nature, buildings, "
    "cartoon characters, video game screenshots, or anything that is NOT a real vehicle or "
    "vehicle part — you must reject it. "
    "Reply with ONLY the single word YES if it is a vehicle or vehicle part, "
    "or the single word NO if it is not. "
    "Do not add any punctuation, explanation, or other words."
)

VLM_PROMPT_TEMPLATE = (
    "You are an expert automotive damage assessor. "
    "You are looking at an annotated vehicle damage photograph. "
    "The AI detection system has identified: {damage_types} affecting approximately {area:,} pixels. "
    "The recommended repair action is: '{repair}'. "
    "In 2-3 concise sentences, describe the visible damage in professional terms — "
    "its location, likely severity, and what a mechanic would need to do. "
    "Write directly, do not start with 'The image shows' or similar. "
    "Do not repeat the repair action verbatim."
)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="DSGP-Group07 Prediction Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the deployed frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class CarSpecRequest(BaseModel):
    Make: str
    Model: str
    YOM: int
    Mileage_km: int
    Engine_cc: int
    Fuel_Type: str
    Gear: str
    Condition: str
    Has_AC: bool
    Has_PowerSteering: bool
    Has_PowerMirror: bool
    Has_PowerWindow: bool

# ---------------------------------------------------------------------------
# Model globals
# ---------------------------------------------------------------------------

prediction_model = None
preprocessing_pipeline = None
yolo_damage_model = None
seg_model = None
seg_processor = None
anomaly_model_bundle = None   # dict: {scaler, model, features, hard_bounds}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_load(name: str, loader_fn):
    """Runs loader_fn(), prints success/failure, returns the result or None."""
    try:
        result = loader_fn()
        print(f"[INFO] {name} loaded seamlessly.")
        return result
    except Exception as e:
        print(f"[ERROR] Could not load {name}: {e}")
        return None


def _to_base64_jpeg(image: Image.Image) -> str:
    """Encodes a PIL Image to a base64 JPEG data URI string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _validate_vehicle_image(image: Image.Image) -> tuple:
    """
    Uses Gemini to confirm the image contains a vehicle or vehicle part.
    Returns (is_valid: bool, gemini_skipped: bool).
    Skips check and returns (True, True) if Gemini is unavailable.
    """
    try:
        model = _get_gemini_model()
        response = model.generate_content([GATEKEEPER_PROMPT, image])
        cleaned = re.sub(r'[^A-Z]', '', response.text.strip().upper() if response else '')
        print(f"[GEMINI GATE] Raw: '{response.text.strip() if response else 'None'}' | Cleaned: '{cleaned}'")
        return cleaned.startswith('YES'), False
    except Exception as e:
        print(f"[GEMINI GATE ERROR] {e}. Skipping gatekeeper.")
        return True, True  # Fail-open + flag that Gemini was skipped


def _run_yolo(image: Image.Image) -> tuple:
    """
    Runs YOLO damage detection on the image.
    Returns (results, has_damage, detected_groups_set, detailed_detections_list).
    """
    results = yolo_damage_model.predict(image)
    detected_groups = set()
    detailed_detections = []
    has_damage = False

    for r in results:
        if len(r.boxes) > 0:
            has_damage = True
            for box, conf, cls_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                class_name = yolo_damage_model.names[int(cls_id)]
                detected_groups.add(class_name)
                detailed_detections.append({
                    "class": class_name,
                    "confidence": float(conf),
                    "box": [float(x) for x in box],
                })

    return results, has_damage, detected_groups, detailed_detections


def _run_segformer(image: Image.Image, img_np: np.ndarray) -> tuple:
    """
    Runs SegFormer segmentation and applies a red damage overlay.
    Falls back to OpenCV edge-contour area estimation if SegFormer is unavailable.
    Returns (annotated_array_rgb, dent_area_pixels, mask_or_None).
    The mask is a boolean H×W numpy array — used downstream to filter YOLO detections.
    """
    seg_annotated = img_np.copy()
    dent_area = 0
    mask = None

    if seg_model is not None and seg_processor is not None:
        inputs = seg_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = seg_model(**inputs)

        upsampled = F.interpolate(
            outputs.logits,
            size=(image.size[1], image.size[0]),
            mode="bilinear",
            align_corners=False,
        )
        # Dynamically pick the damage channel:
        #   Binary model [1, 2, H, W] → channel 1 (damage)
        #   Single-class  [1, 1, H, W] → channel 0
        damage_channel = 1 if upsampled.shape[1] > 1 else 0
        mask = (torch.sigmoid(upsampled[0, damage_channel]) > 0.5).numpy()
        dent_area = int(np.sum(mask))
        seg_annotated[mask] = (seg_annotated[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

    else:
        # Fallback: largest contour bounding-box area via OpenCV
        img_bgr = img_np[:, :, ::-1].copy()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            dent_area = w * h

    return seg_annotated, dent_area, mask


def _filter_detections_by_mask(detailed_detections: list, mask) -> tuple:
    """
    Filters YOLO detections to only those whose bounding box has ≥20% overlap
    with the SegFormer damage mask. Eliminates false positives scattered
    across the image when the real damage is in a specific region.

    Returns (filtered_detected_groups_set, filtered_detailed_detections_list).
    Falls back to all detections if mask is None (SegFormer unavailable).
    """
    if mask is None or not detailed_detections:
        return {d['class'] for d in detailed_detections}, detailed_detections

    h, w = mask.shape
    filtered = []
    for det in detailed_detections:
        x1, y1, x2, y2 = [int(v) for v in det['box']]
        # Clamp coordinates to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area == 0:
            continue
        overlap_pixels = int(np.sum(mask[y1:y2, x1:x2]))
        if overlap_pixels / box_area >= 0.20:  # at least 20% of box inside damage region
            filtered.append(det)

    if not filtered:
        # No YOLO box overlapped the mask — SegFormer found something YOLO missed entirely.
        # Return empty so the caller can handle the SegFormer-only case.
        return set(), []

    return {d['class'] for d in filtered}, filtered


def _estimate_repair(dent_area: int, total_pixels: int) -> tuple:
    """Maps SegFormer pixel area to a repair action and cost.
    Uses percentage of image area so results are resolution-independent."""
    damage_pct = (dent_area / total_pixels * 100) if total_pixels > 0 else 0
    for max_pct, repair, cost in REPAIR_TIERS:
        if max_pct is None or damage_pct < max_pct:
            return repair, cost, round(damage_pct, 2)
    return REPAIR_TIERS[-1][1], REPAIR_TIERS[-1][2], round(damage_pct, 2)


def _generate_vlm_description(annotated_image: Image.Image, damage_types: list, dent_area: int, repair: str) -> tuple:
    """
    Asks Gemini to generate a professional damage description from the annotated image.
    Returns (description: str, gemini_skipped: bool).
    Returns fallback template + skipped=True if the API call fails.
    """
    try:
        model = _get_gemini_model()
        prompt = VLM_PROMPT_TEMPLATE.format(
            damage_types=', '.join(damage_types),
            area=dent_area,
            repair=repair,
        )
        response = model.generate_content([prompt, annotated_image])
        description = response.text.strip() if response else None
        print(f"[GEMINI VLM] Description generated ({len(description or '')} chars)")
        if description:
            return description, False
    except Exception as e:
        print(f"[GEMINI VLM ERROR] {e}. Using fallback description.")

    # Fallback template
    fallback = (
        f"Vehicle shows distinct damage ({', '.join(damage_types)}). "
        f"Structural integrity assessment suggests {dent_area:,}px affected area. "
        f"The estimated market repair cost in Sri Lanka leans towards '{repair}' via local suppliers."
    )
    return fallback, True

# ---------------------------------------------------------------------------
# Startup: load all ML models
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_models():
    global prediction_model, preprocessing_pipeline, yolo_damage_model, seg_model, seg_processor, anomaly_model_bundle

    model_path    = os.path.join(ROOT_DIR, "price-model", "best_optimized_model.pkl")
    preproc_path  = os.path.join(ROOT_DIR, "price-model", "preprocessing_optimized.pkl")
    anomaly_path  = os.path.join(ROOT_DIR, "price-model", "anomaly_model.pkl")
    yolo_path     = os.path.join(ROOT_DIR, "damage-detection", "models", "v2.pt")
    seg_path      = os.path.join(ROOT_DIR, "damage-detection", "models", "best_model")

    def _load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    prediction_model       = _safe_load("XGBoost Price Model",      lambda: _load_pickle(model_path))
    preprocessing_pipeline = _safe_load("Preprocessing Pipeline",   lambda: _load_pickle(preproc_path))
    anomaly_model_bundle   = _safe_load("Anomaly Detection Model",  lambda: _load_pickle(anomaly_path))

    if os.path.exists(yolo_path):
        yolo_damage_model = _safe_load("YOLO Damage Model", lambda: YOLO(yolo_path))
    else:
        print(f"[WARNING] YOLO model not found at {yolo_path}")

    if os.path.exists(seg_path) and os.listdir(seg_path):
        seg_processor = _safe_load("SegFormer Processor", lambda: SegformerImageProcessor.from_pretrained(seg_path))
        seg_model     = _safe_load("SegFormer Model",     lambda: SegformerForSemanticSegmentation.from_pretrained(seg_path))
    else:
        print(f"[WARNING] SegFormer model directory empty or not found at {seg_path}")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/vehicle_options")
def get_vehicle_options():
    csv_path = os.path.join(ROOT_DIR, "data", "initial-cleaning", "cleaned_no_outliers.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=500, detail="Vehicle dataset not found.")
    try:
        df = pd.read_csv(csv_path)
        mapping = df.groupby('Make')['Model'].apply(lambda x: sorted(set(x))).to_dict()
        return {"status": "success", "options": mapping}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_price")
def predict_car_value(specs: CarSpecRequest):
    if prediction_model is None or preprocessing_pipeline is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    option_count = sum([specs.Has_AC, specs.Has_PowerSteering, specs.Has_PowerMirror, specs.Has_PowerWindow])

    input_data = {
        'Make': specs.Make, 'Model': specs.Model, 'YOM': specs.YOM,
        'Mileage (km)': specs.Mileage_km, 'Engine (cc)': specs.Engine_cc,
        'Fuel Type': specs.Fuel_Type, 'Gear': specs.Gear, 'Condition': specs.Condition,
        'Option_Count': option_count,
        'Has_AC': int(specs.Has_AC), 'Has_PowerSteering': int(specs.Has_PowerSteering),
        'Has_PowerMirror': int(specs.Has_PowerMirror), 'Has_PowerWindow': int(specs.Has_PowerWindow),
    }

    # --- Anomaly Detection on user input ---
    # Two-layer check: hard bounds (immediate reject) + OneClassSVM (statistical)
    is_anomalous = False
    anomaly_reason = None

    if anomaly_model_bundle is not None:
        hard_bounds = anomaly_model_bundle.get("hard_bounds", {})

        # Layer 1: Hard bounds — catch obviously impossible values
        checks = [
            ("Engine (cc)",  specs.Engine_cc,  "Engine size"),
            ("Mileage (km)", specs.Mileage_km, "Mileage"),
            ("YOM",          specs.YOM,         "Year of manufacture"),
        ]
        for field, value, label in checks:
            if field in hard_bounds:
                lo, hi = hard_bounds[field]
                if not (lo <= value <= hi):
                    is_anomalous = True
                    anomaly_reason = f"{label} value {value:,} is outside the plausible range ({lo:,}–{hi:,})."
                    print(f"[ANOMALY] Hard bound triggered: {anomaly_reason}")
                    break

        # Layer 2: OneClassSVM statistical check (only if hard bounds passed)
        if not is_anomalous:
            try:
                import pandas as pd
                scaler   = anomaly_model_bundle["scaler"]
                oc_svm   = anomaly_model_bundle["model"]
                features = anomaly_model_bundle["features"]   # ["Price", "Mileage (km)", "Engine (cc)"]
                # Use Brand_Avg_Price as a proxy for Price since we don't have actual price yet
                # We scale with 0 for Price so SVM only weighs Mileage + Engine statistically
                probe = pd.DataFrame([{
                    "Price":        0,   # unknown at input time — neutral
                    "Mileage (km)": specs.Mileage_km,
                    "Engine (cc)":  specs.Engine_cc,
                }])[features]
                scaled = scaler.transform(probe)
                svm_pred = oc_svm.predict(scaled)[0]
                if svm_pred == -1:
                    is_anomalous = True
                    anomaly_reason = "The combination of mileage and engine size appears statistically unusual for Sri Lankan vehicles."
                    print(f"[ANOMALY] SVM flagged input: mileage={specs.Mileage_km}, engine={specs.Engine_cc}")
            except Exception as e:
                print(f"[ANOMALY CHECK ERROR] {e}. Skipping SVM check.")
    else:
        print("[WARNING] Anomaly model not loaded — skipping input validation.")

    try:
        prediction, _ = predict_price(input_data, prediction_model, preprocessing_pipeline)
        if prediction is None:
            raise HTTPException(status_code=400, detail="Prediction failed due to invalid features.")
        return {
            "status": "success",
            "predicted_price_lkr": int(prediction),
            "is_anomalous": is_anomalous,
            "anomaly_reason": anomaly_reason,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_damage")
async def analyze_damage_chain(file: UploadFile = File(...)):
    """
    Gatekeeper → YOLO detection → SegFormer segmentation →
    Annotation → Cost estimation → Gemini VLM description.
    """
    if yolo_damage_model is None:
        raise HTTPException(status_code=500, detail="YOLO Model not loaded.")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_np = np.array(image)

        # Step 0: Gatekeeper — reject non-vehicle images before heavy inference
        is_valid, gate_skipped = _validate_vehicle_image(image)
        if not is_valid:
            return {"status": "invalid_image", "message": "This does not appear to be a photo of a vehicle! Please upload a valid car image."}

        # Step 1: SegFormer segmentation + red overlay (runs FIRST — defines the damage region)
        seg_annotated, dent_area, seg_mask = _run_segformer(image, img_np)

        # Step 2: YOLO detection — runs on full image but results filtered by SegFormer mask
        results, yolo_has_damage, all_detected_groups, all_detections = _run_yolo(image)
        detected_groups, detailed_detections = _filter_detections_by_mask(all_detections, seg_mask)
        has_damage = bool(detected_groups)  # True only if a YOLO box overlaps the SegFormer mask

        # Step 3: Output image — SegFormer red overlay ONLY (no YOLO bounding boxes)
        # This prevents noisy/scattered YOLO boxes from cluttering the result photo
        annotated_image = Image.fromarray(seg_annotated)  # pure SegFormer mask, clean
        image_uri = _to_base64_jpeg(annotated_image)

        # Dual-signal damage verdict:
        #   YOLO   = structural damage (dents, broken panels) — bounding box signal
        #   SegFormer = surface damage (scratches, paint) — pixel-area signal
        # Damage is confirmed if EITHER model fires.
        segformer_has_damage = dent_area >= MIN_SEGFORMER_DAMAGE_PX
        effective_has_damage = has_damage or segformer_has_damage

        if not effective_has_damage:
            return {"status": "success", "has_damage": False, "image": image_uri, "message": "No visible damage detected by YOLO or SegFormer."}

        # Step 4: Build damage group list — if YOLO missed it, label SegFormer's finding explicitly
        detected_groups_list = sorted(detected_groups)
        if not has_damage and segformer_has_damage:
            detected_groups_list = ["scratches"]
            print(f"[INFO] YOLO found no bounding boxes, but SegFormer detected {dent_area:,}px of scratch-level surface damage.")

        # Step 5: Repair cost estimation (driven by SegFormer pixel area as % of image)
        total_pixels = image.size[0] * image.size[1]
        repair, cost, damage_pct = _estimate_repair(dent_area, total_pixels)

        # Step 6: Gemini VLM damage description
        vlm_reasoning, vlm_skipped = _generate_vlm_description(annotated_image, detected_groups_list, dent_area, repair)
        gemini_skipped = gate_skipped or vlm_skipped

        return {
            "status": "success",
            "has_damage": True,
            "image": image_uri,
            "detected_groups": detected_groups_list,
            "vlm_reasoning": vlm_reasoning,
            "estimated_cost_lkr": cost,
            "repair_action": repair,
            "damage_pct": damage_pct,
            "detections": detailed_detections,
            "gemini_skipped": gemini_skipped,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
