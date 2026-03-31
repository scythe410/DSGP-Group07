import streamlit as st
import time
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Import models
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO


# Model Loading (Cached to save memory and time)
@st.cache_resource
def load_segformer():
    # saved model in the hugging face format
    model_path = "best_model"
    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    return processor, model


@st.cache_resource
def load_yolo():
    return YOLO("models/v2.pt")


# UI Setup
st.set_page_config(layout="wide")
st.title("Damage Detection Comparison")
st.write("Compare YOLO (Bounding Boxes) vs. SegFormer (Exact Pixel Segmentation)")

# initialize session state for both outputs
if "yolo_image" not in st.session_state:
    st.session_state.yolo_image = None
if "segformer_image" not in st.session_state:
    st.session_state.segformer_image = None

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.yolo_image = None
        st.session_state.segformer_image = None
        st.session_state.last_uploaded = uploaded_file.name

ui_container = st.empty()
initial_image_placeholder = st.empty()

if uploaded_file is None:
    initial_image_placeholder.markdown(
        """
        <div style='width:100%;height:400px;border:3px dashed grey;
                    display:flex;align-items:Center;justify-content:center;'>
            <p style='font-size:18px;color:grey;'>Upload Image to Compare Models</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    img = Image.open(uploaded_file).convert("RGB")

    # Show button only if images have not been processed yet
    if st.session_state.yolo_image is None or st.session_state.segformer_image is None:

        # Display original image before processing
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            initial_image_placeholder.image(img, use_container_width=True, caption="Original Image")

        # add button in the UI container
        if ui_container.button("Run Comparison", use_container_width=True):
            initial_image_placeholder.empty()  # Remove original image to prep for side-by-side
            progress = st.progress(0, text="Loading models...")

            # Load models
            seg_processor, seg_model = load_segformer()
            yolo_model = load_yolo()
            progress.progress(20, text="Models loaded. Running YOLO inference...")

            # YOLO Inference
            results = yolo_model(img)
            annotated_bgr = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.session_state.yolo_image = Image.fromarray(annotated_rgb)

            progress.progress(50, text="YOLO complete. Running SegFormer inference...")

            # SegFormer Inference
            inputs = seg_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = seg_model(**inputs)

            # Resize mask properly
            logits = outputs.logits
            upsampled_logits = F.interpolate(
                logits,
                size=(img.size[1], img.size[0]),
                mode="bilinear",
                align_corners=False
            )

            probs = torch.sigmoid(upsampled_logits).squeeze()
            mask = (probs > 0.5).numpy()

            # Create red overlay
            img_np = np.array(img)
            seg_annotated = img_np.copy()
            seg_annotated[mask] = (seg_annotated[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
            st.session_state.segformer_image = Image.fromarray(seg_annotated)

            progress.progress(100, text="Processing complete!")
            time.sleep(0.5)

            ui_container.empty()
            st.rerun()

    else:
        # Display the side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("YOLO v8")
            st.image(st.session_state.yolo_image, use_container_width=True, caption="Bounding Box Detection")

        with col2:
            st.subheader("SegFormer")
            st.image(st.session_state.segformer_image, use_container_width=True, caption="Pixel Segmentation Overlay")

        if st.button("Clear Results"):
            st.session_state.yolo_image = None
            st.session_state.segformer_image = None
            st.rerun()


