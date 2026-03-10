from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import pickle
import pandas as pd
import io
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import base64
from huggingface_hub import hf_hub_download

# Add root so we can import models from other folders cleanly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'price-model'))
sys.path.append(os.path.join(ROOT_DIR, 'data', 'pipeline'))

from predictor import predict_price

app = FastAPI(title="DSGP-Group07 Prediction Backend API")

# Allow Frontend to talk to this Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Pydantic Model for Input Validation (Ensures Frontend sends right data)
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

# Load Models once at startup
prediction_model = None
preprocessing_pipeline = None
yolo_damage_model = None
xgboost_load_error = None

@app.on_event("startup")
def load_models():
    global prediction_model, preprocessing_pipeline, yolo_damage_model, xgboost_load_error
    try:
        print("[INFO] Downloading XGBoost models from Hugging Face Hub...")
        model_path = hf_hub_download(repo_id="scythe410/sri-lankan-vehicle-price-prediction", filename="best_optimized_model.pkl")
        preproc_path = hf_hub_download(repo_id="scythe410/sri-lankan-vehicle-price-prediction", filename="preprocessing_optimized.pkl")
        
        with open(model_path, 'rb') as f:
            prediction_model = pickle.load(f)
        with open(preproc_path, 'rb') as f:
            preprocessing_pipeline = pickle.load(f)
        print("[INFO] XGBoost Price Model loaded seamlessly.")
    except Exception as e:
        xgboost_load_error = str(e)
        print(f"[ERROR] Could not load XGBoost model: {e}")
        
    try:
        print("[INFO] Downloading YOLO damage model from Hugging Face Hub...")
        yolo_path = hf_hub_download(repo_id="scythe410/vehicle-damage-detection-yolo", filename="v2.pt")
        yolo_damage_model = YOLO(yolo_path)
        print("[INFO] YOLO Damage Model loaded seamlessly.")
    except Exception as e:
        print(f"[ERROR] Could not load YOLO model: {e}")

@app.post("/predict_price")
def predict_car_value(specs: CarSpecRequest):
    if prediction_model is None or preprocessing_pipeline is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    # 1. Anomaly Checking (Ramiru's Pipeline Integration)
    # TODO: Connect Ramiru's One-Class SVM here to block garbage data before predicting

    # 2. Format data for Osanda's XGBoost Predictor
    option_count = sum([specs.Has_AC, specs.Has_PowerSteering, specs.Has_PowerMirror, specs.Has_PowerWindow])
    
    input_data = {
        'Make': specs.Make,
        'Model': specs.Model,
        'YOM': specs.YOM,
        'Mileage (km)': specs.Mileage_km,
        'Engine (cc)': specs.Engine_cc,
        'Fuel Type': specs.Fuel_Type,
        'Gear': specs.Gear,
        'Condition': specs.Condition,
        'Option_Count': option_count,
        'Has_AC': 1 if specs.Has_AC else 0,
        'Has_PowerSteering': 1 if specs.Has_PowerSteering else 0,
        'Has_PowerMirror': 1 if specs.Has_PowerMirror else 0,
        'Has_PowerWindow': 1 if specs.Has_PowerWindow else 0
    }

    try:
        # Run XGBoost Inference
        prediction, df_features = predict_price(input_data, prediction_model, preprocessing_pipeline)
        
        if prediction is not None:
            return {
                "status": "success",
                "predicted_price_lkr": int(prediction),
                "is_anomalous": False, # Placeholder until SVM is connected
                "confidence_score": 0.95 # Placeholder
            }
        else:
             raise HTTPException(status_code=400, detail="Prediction failed due to invalid features.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_damage")
async def analyze_damage_chain(file: UploadFile = File(...)):
    """
    1. Receives image from Frontend
    2. Runs Dewlini's YOLO Model
    3. Chains output to Savindi's VLM Logic (w/ local RAM safety)
    """
    if yolo_damage_model is None:
        raise HTTPException(status_code=500, detail="YOLO Model not loaded.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Step 1: YOLO Detection (In memory to avoid Live Server refresh loops)
        results = yolo_damage_model.predict(image, conf=0.1)
        
        detected_groups = []
        has_damage = False
        plotted_image_b64 = None
        for r in results:
            if len(r.boxes) > 0:
                has_damage = True
                for cls_id in r.boxes.cls:
                    class_name = yolo_damage_model.names[int(cls_id)]
                    if class_name not in detected_groups:
                        detected_groups.append(class_name)
        
        if has_damage:
            # Generate image with bounding boxes, encode via OpenCV
            im_bgr = results[0].plot()
            _, buffer = cv2.imencode('.jpg', im_bgr)
            plotted_image_b64 = base64.b64encode(buffer).decode('utf-8')
                        
        if not has_damage:
             return {
                 "status": "success", 
                 "has_damage": False,
                 "message": "No visible damage detected by YOLO."
             }
             
        # Step 2: VLM & OpenCV Heuristics Pipeline
        # Convert to OpenCV format
        open_cv_image = np.array(image) 
        # Convert RGB to BGR 
        img = open_cv_image[:, :, ::-1].copy() 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dent_area = 0
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(largest)
            dent_area = w * h
            
            if dent_area < 5000:
                dent_type = "Small Dent"
            # Mock VLM / Real VLM Logic Extraction
        vlm_reasoning_text = ""
        cost = 0
        repair = "None required"
        
        if "dent" in detected_groups or "severe-dent" in detected_groups:
             cost += 25000
             repair = "Panel pull and repaint"
        if "scratch" in detected_groups:
             cost += 5000
             repair = "Buff / Polish / Spot repaint"
             
        # Mock VLM heuristic reasoning (Fallback)
        vlm_mock_text = f"Based on the YOLO bounding boxes provided, the vehicle shows visible damage ({', '.join(detected_groups)}). The estimated market repair cost in Sri Lanka leans towards an action of '{repair}' costing LKR {cost:,} via local suppliers."
        vlm_reasoning_text = vlm_mock_text

        # Attempt REAL Gemini VLM inference if API key exists
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key and plotted_image_b64:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                # Ensure the model is available. 
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # We need to construct a PIL image from the base64 string to feed into gemini
                image_data = base64.b64decode(plotted_image_b64)
                pil_img = Image.open(io.BytesIO(image_data))
                
                vlm_prompt = f"You are a sophisticated automotive damage assessor VLM. Look at the provided image with YOLO bounding boxes. We have mathematically calculated a base repair cost of LKR {cost} requiring '{repair}'. Do you agree with this assessment? If so, why? Provide a 2-3 sentence technical diagnostic reasoning explaining the severity of the damage visible in the colored boxes to the user."
                
                response = model.generate_content([vlm_prompt, pil_img])
                vlm_reasoning_text = response.text
            except Exception as e:
                print(f"[ERROR] Gemini VLM logic failed, falling back to heuristic: {e}")
        
        return {
            "status": "success",
            "has_damage": True,
            "detected_groups": detected_groups,
            "vlm_reasoning": vlm_reasoning_text,
            "estimated_cost_lkr": cost,
            "repair_action": repair,
            "sides_affected": ["Front", "Left side"], # Static for demo, could map from YOLO boxes
            "image_base64": plotted_image_b64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "message": "Backend API is humming nicely",
        "xgboost_load_error": xgboost_load_error
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
