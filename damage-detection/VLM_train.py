"""NOTE: Use GPU for inferencing, use pip install to install the necessary packages"""

"""
SETUP INSTRUCTIONS:
-> Replace "add_your_token" with your HF token {Hugging Face token}
-> Replace 'path_to_yolo_model' with your YOLO model path
-> Replace "add_the_img_file_to_run" with your image path you want to test with
-> Requires GPU, pip install torch ultralytics transformers opencv-python pillow matplotlib
"""

import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import matplotlib.pyplot as plt

# hugging face token for authentication
HF_TOKEN = "add_your_token"
from huggingface_hub import login
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
else:
    login()

# model loading
model_id = "google/paligemma2-3b-mix-448"
processor = PaliGemmaProcessor.from_pretrained(model_id)
vlm_model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

yolo_model = YOLO('path_to_yolo_model')

# cost lookup table
COST_TABLE = {
    "scratch": {
        "clear coat":  ("$50–$150",   "Rs. 15,000–45,000",   "Light polish or clear coat respray only."),
        "paint level": ("$150–$400",  "Rs. 45,000–120,000",  "Colour coat respray required for affected panel."),
        "deep":        ("$400–$900",  "Rs. 120,000–270,000", "Bare metal treatment, primer, full respray needed."),
        "bare metal":  ("$400–$900",  "Rs. 120,000–270,000", "Bare metal treatment, primer, full respray needed."),
    },
    "dent": {
        "paintless":   ("$75–$200",   "Rs. 22,000–60,000",   "PDR applicable — no paint damage, smooth dent."),
        "traditional": ("$200–$600",  "Rs. 60,000–180,000",  "Filler, sanding and respray required."),
        "severe":      ("$600–$2000", "Rs. 180,000–600,000", "Panel replacement or major bodywork likely needed."),
        "creasing":    ("$600–$2000", "Rs. 180,000–600,000", "Panel replacement or major bodywork likely needed."),
    }
}

PART_MULTIPLIERS = {
    "hood":    1.4,
    "bonnet":  1.4,
    "door":    1.2,
    "bumper":  1.0,
    "fender":  1.3,
    "quarter": 1.5,
    "roof":    1.6,
    "trunk":   1.2,
    "boot":    1.2,
}

def run_vlm_query(prompt, image, max_tokens=50):
    """Helper to run a single VQA-style query and return the answer."""
    full_prompt = f"<image>{prompt}"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = vlm_model.generate(**inputs, max_new_tokens=max_tokens)
    answer = processor.decode(output[0], skip_special_tokens=True).split(prompt)[-1].strip()
    return answer.lower()


def apply_nms(boxes, scores, iou_threshold=0.4):
    """
    Custom NMS to remove overlapping YOLO detections.
    boxes  : list/tensor of [x1, y1, x2, y2]
    scores : list/tensor of confidence scores
    Returns indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    boxes  = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort by confidence score descending
    order = scores.argsort()[::-1]
    keep  = []

    while order.size > 0:
        idx = order[0]
        keep.append(idx)

        # Compute IoU of current box with all remaining boxes
        ix1 = np.maximum(x1[idx], x1[order[1:]])
        iy1 = np.maximum(y1[idx], y1[order[1:]])
        ix2 = np.minimum(x2[idx], x2[order[1:]])
        iy2 = np.minimum(y2[idx], y2[order[1:]])

        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        inter   = inter_w * inter_h

        iou = inter / (areas[idx] + areas[order[1:]] - inter)

        # Keep only boxes with IoU below threshold
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def get_cost_estimate(damage_type, detail, car_part):
    """Build cost estimate from lookup table + part multiplier."""
    cost_usd  = None
    cost_lkr  = None
    repair_note = None
    category  = "scratch" if "scratch" in damage_type else "dent" if "dent" in damage_type else None

    if category:
        for key in COST_TABLE[category]:
            if key in detail:
                cost_usd, cost_lkr, repair_note = COST_TABLE[category][key]
                break
        if not cost_usd:
            default = "paint level" if category == "scratch" else "traditional"
            cost_usd, cost_lkr, repair_note = COST_TABLE[category][default]
    else:
        cost_usd    = "$100–$500"
        cost_lkr    = "Rs. 30,000–150,000"
        repair_note = "Mixed damage — professional assessment recommended."

    part_note = ""
    for part, mult in PART_MULTIPLIERS.items():
        if part in car_part:
            if mult >= 1.4:
                part_note = f"Note: {car_part.title()} is a complex panel — costs may be on the higher end."
            break

    return cost_usd, cost_lkr, repair_note, part_note


def build_summary(car_part, paint_finish, damage_type, detail,
                  cost_usd, cost_lkr, repair_note, part_note):
    """Build a clean human-readable report."""
    summary = (
        f"Damage identified on the {car_part.title()}. "
        f"Paint finish appears to be {paint_finish}. "
    )
    if "scratch" in damage_type:
        summary += f"Scratch classified as: {detail}. "
    elif "dent" in damage_type:
        summary += f"Dent classified as: {detail}. "
    else:
        summary += f"Damage detail: {detail}. "

    summary += (
        f"Recommended repair: {repair_note} "
        f"Estimated cost: {cost_usd} / {cost_lkr}. "
    )
    if part_note:
        summary += part_note
    return summary


def process_full_image_analysis(image_path):
    # 1. Load original image
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # 2. YOLO Inference
    yolo_results = yolo_model.predict(img_rgb, conf=0.4)
    if not yolo_results[0].boxes:
        print("No damage detected above 0.4 confidence.")
        return

    raw_boxes  = yolo_results[0].boxes.xyxy.cpu().numpy()
    raw_scores = yolo_results[0].boxes.conf.cpu().numpy()

    print(f"Raw detections from YOLO: {len(raw_boxes)}")

    # 3. Apply custom NMS to remove overlapping detections
    keep_indices = apply_nms(raw_boxes, raw_scores, iou_threshold=0.4)
    filtered_boxes  = raw_boxes[keep_indices]
    filtered_scores = raw_scores[keep_indices]

    print(f"After removing overlaps  : {len(filtered_boxes)} unique damage zone(s) remaining.")

    # 4. Loop through filtered boxes only
    for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
        annotated_img = img_rgb.copy()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 6)
        final_input_img = Image.fromarray(annotated_img)

        # --- Queries that always run ---
        car_part = run_vlm_query(
            "answer en what car part is damaged inside the blue box?",
            final_input_img)

        paint_finish = run_vlm_query(
            "answer en what is the paint finish of the car inside the blue box: standard, metallic, pearl, or matte?",
            final_input_img)

        damage_type = run_vlm_query(
            "answer en is the damage inside the blue box a scratch or a dent?",
            final_input_img)

        # --- Conditional Query based on damage type ---
        if "scratch" in damage_type:
            detail = run_vlm_query(
                "answer en is the scratch inside the blue box a clear coat scratch, paint level scratch, or deep scratch with bare metal?",
                final_input_img)
            detail_label = "Scratch Depth"

        elif "dent" in damage_type:
            detail = run_vlm_query(
                "answer en is the dent inside the blue box eligible for paintless dent repair, traditional repair, or is it severe with creasing?",
                final_input_img)
            detail_label = "Dent Type"

        else:
            detail = run_vlm_query(
                "answer en describe the damage type inside the blue box in one word.",
                final_input_img)
            detail_label = "Damage Detail"

        # --- Build Cost Estimate and Summary ---
        cost_usd, cost_lkr, repair_note, part_note = get_cost_estimate(
            damage_type, detail, car_part)
        summary = build_summary(car_part, paint_finish, damage_type, detail,
                                cost_usd, cost_lkr, repair_note, part_note)

        # 5. Print Report
        print(f"\n{'='*55}")
        print(f"  Damage Zone #{i+1}  |  YOLO Confidence: {score:.2f}")
        print(f"{'='*55}")
        print(f"  Car Part      : {car_part.title()}")
        print(f"  Paint Finish  : {paint_finish.title()}")
        print(f"  Damage Type   : {damage_type.title()}")
        print(f"  {detail_label:<14}: {detail.title()}")
        print(f"  Cost (USD)    : {cost_usd}")
        print(f"  Cost (LKR)    : {cost_lkr}")
        print(f"\n  Summary: {summary}")
        print(f"{'='*55}")

        plt.figure(figsize=(8, 8))
        plt.imshow(annotated_img)
        plt.title(f"Zone #{i+1} | {car_part.title()} | {cost_usd} / {cost_lkr}", fontsize=9)
        plt.axis('off')
        plt.show()

# Run it
process_full_image_analysis("add_the_img_file_to_run")




