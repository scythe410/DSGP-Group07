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
import base64
from ultralytics import YOLO

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

@app.on_event("startup")
def load_models():
    global prediction_model, preprocessing_pipeline, yolo_damage_model
    try:
        model_path = os.path.join(ROOT_DIR, "price-model", "best_optimized_model.pkl")
        preproc_path = os.path.join(ROOT_DIR, "price-model", "preprocessing_optimized.pkl")
        
        with open(model_path, 'rb') as f:
            prediction_model = pickle.load(f)
        with open(preproc_path, 'rb') as f:
            preprocessing_pipeline = pickle.load(f)
        print("[INFO] XGBoost Price Model loaded seamlessly.")
    except Exception as e:
        print(f"[ERROR] Could not load XGBoost model: {e}")
        
    try:
        yolo_path = os.path.join(ROOT_DIR, "damage-detection", "models", "v2.pt")
        if os.path.exists(yolo_path):
            yolo_damage_model = YOLO(yolo_path)
            print("[INFO] YOLO Damage Model loaded seamlessly.")
        else:
             print(f"[WARNING] YOLO model not found at {yolo_path}")
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
        results = yolo_damage_model.predict(image)
        
        detected_groups = []
        has_damage = False
        for r in results:
            if len(r.boxes) > 0:
                has_damage = True
                for cls_id in r.boxes.cls:
                    class_name = yolo_damage_model.names[int(cls_id)]
                    if class_name not in detected_groups:
                        detected_groups.append(class_name)
                        
        # Render the annotated image with bounding boxes
        plotted_img_array = results[0].plot()
        # Convert BGR (OpenCV) to RGB (PIL)
        plotted_img_rgb = plotted_img_array[..., ::-1] 
        annotated_image = Image.fromarray(plotted_img_rgb)
        
        # Base64 encode the resulting image
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
        if not has_damage:
             return {
                 "status": "success", 
                 "has_damage": False,
                 "image": f"data:image/jpeg;base64,{encoded_image}",
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
        repair = "Inspection required"
        cost = 0
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(largest)
            dent_area = w * h
            
            if dent_area < 5000:
                dent_type = "Small Dent"
                repair = "Paintless Dent Repair"
                cost = 5000
            elif dent_area < 15000:
                dent_type = "Medium Dent"
                repair = "Panel Beating"
                cost = 12000
            else:
                dent_type = "Severe Structure Damage"
                repair = "Panel Replacement + Repaint"
                cost = 35000
                
        # Simulate VLM Text Generation (to prevent 15GB RAM crash locally)
        vlm_mock_text = f"Based on the bounding boxes provided, the vehicle shows distinct damage ({', '.join(detected_groups)}). Structural integrity assessment suggests {dent_area:,}px affected area. The estimated market repair cost in Sri Lanka leans towards '{repair}' via local suppliers."
        
        return {
            "status": "success",
            "has_damage": True,
            "image": f"data:image/jpeg;base64,{encoded_image}",
            "detected_groups": detected_groups,
            "vlm_reasoning": vlm_mock_text,
            "estimated_cost_lkr": cost,
            "repair_action": repair,
            "sides_affected": ["Front", "Left side"] # Static for demo, could map from YOLO boxes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend API is humming nicely"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
