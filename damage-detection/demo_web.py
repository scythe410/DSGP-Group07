import streamlit as st
import time
from PIL import Image
from ultralytics import YOLO
import cv2

st.title("Damage Detection")

# initialize session state
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# reset if a new file uploaded
if uploaded_file:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.processed_image = None
        st.session_state.last_uploaded = uploaded_file.name

# creating a container for the UI elements needed to remove
ui_container = st.empty()
image_placeholder = st.empty()

if uploaded_file is None:
    image_placeholder.markdown(
        """
        <div style='width:600px;height:600px;border:3px dashed grey;
                    display:flex;align-items:Center;justify-content:center;'>
            <p style='font-size:18px;color:grey;'>Upload Image</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    img = Image.open(uploaded_file)

    # show button only if not processed
    if st.session_state.processed_image is None:
        image_placeholder.image(img, width=600)

        # add button in the UI container
        if ui_container.button("Process Image"):
            progress = st.progress(0)
            for i in range(1,101):
                time.sleep(0.01)
                progress.progress(i)

            model = YOLO("models/v2.pt")
            results = model(img)
            annotated_bgr = results[0].plot()

            # convert to rgb
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            # clear UI container
            st.session_state.processed_image = Image.fromarray(annotated_rgb)
            ui_container.empty()
            st.rerun()
    else:
        # display processed image
        image_placeholder.image(st.session_state.processed_image, width=600)