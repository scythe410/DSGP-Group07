!pip install -q transformers accelerate torch torchvision pillow


import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-flan-t5-xl"
)

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16,
    device_map="auto"
)


from google.colab import drive
drive.mount('/content/drive')


prompt = """
You are an automotive damage assessment assistant.

You are given:
• A cropped image of a vehicle damage
• The detected damage type from an object detection model
• The damage area ratio relative to the full image

Your task:
1. Assess the severity of the damage.
2. Decide the required repair action.

Severity must be one of:
- Minor
- Moderate
- Severe

Repair action must be one of:
Dent:
- Dent pull only
- Dent pull + repaint
- Panel repair or replacement

Scratch:
- Polish only
- Repaint
- Fill and repaint

Damage type: dent
Area ratio: 0.037

Respond ONLY in JSON format:
{
  "severity": "<Minor | Moderate | Severe>",
  "repair_action": "<one valid action>",
  "confidence": "<0.0 – 1.0>"
}
"""


inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt"
).to(device, torch.float16)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=200
)

output = processor.decode(
    generated_ids[0],
    skip_special_tokens=True
)

print(output)


import cv2
import numpy as np
from google.colab import files
from matplotlib import pyplot as plt

# upload image
uploaded = files.upload()

img_path = list(uploaded.keys())[0]

# read image
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur to remove noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# detect edges
edges = cv2.Canny(blur, 50, 150)

plt.imshow(edges, cmap='gray')
plt.title("Detected Edges")
plt.show()

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# choose largest contour (assume dent)
largest = max(contours, key=cv2.contourArea)

x,y,w,h = cv2.boundingRect(largest)

dent = img[y:y+h, x:x+w]

plt.imshow(cv2.cvtColor(dent, cv2.COLOR_BGR2RGB))
plt.title("Detected Dent Area")
plt.show()

dent_area = w * h
print("Dent area:", dent_area)

if dent_area < 5000:
    dent_type = "Small Dent"
    repair = "Paintless Dent Repair"
    cost = 5000

elif dent_area < 15000:
    dent_type = "Medium Dent"
    repair = "Panel Beating"
    cost = 12000

else:
    dent_type = "Severe Dent"
    repair = "Panel Replacement + Paint"
    cost = 25000

print("Dent Type:", dent_type)
print("Repair Needed:", repair)
print("Estimated Cost:", cost, "LKR")
