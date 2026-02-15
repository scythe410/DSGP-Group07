
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
from PIL import Image
from ultralytics import YOLO

# --- Path Setup ---
# Get the absolute path to the repository root
# Current file is in app/app.py, so root is two levels up
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add subdirectories to path for imports
sys.path.append(os.path.join(ROOT_DIR, 'price-model'))
sys.path.append(os.path.join(ROOT_DIR, 'damage-detection'))

from predictor import predict_price

# --- Configuration & Design ---
st.set_page_config(
    page_title="AutoAnalyze Pro",
    page_icon="[PRC DMG EVL SYS]",
    layout="wide"
)

# Custom Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');

    /* Variables */
    :root {
        --main-surface: #E6E5E1;
        --primary-text: #111111;
        --action-accent: #FF5C1A;
        --secondary-ui: #999895;
    }

    /* Global Reset */
    .stApp {
        background-color: var(--main-surface);
        font-family: 'Manrope', sans-serif;
        color: var(--primary-text);
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Manrope', sans-serif;
        font-weight: 600;
        color: var(--primary-text);
        letter-spacing: -0.5px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        color: var(--secondary-ui);
        font-weight: 500;
        font-size: 16px;
        border-bottom: 2px solid transparent;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary-text);
        border-bottom: 2px solid var(--action-accent);
    }

    /* Cards/Containers */
    .css-1r6slb0, .stFileUploader, .stForm {
        background-color: #F0EFEC; /* Slightly lighter shade for cards */
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #DCDBD8;
    }

    /* Labels - High Specificity for Widget Titles (Make, Model, etc.) */
    .stWidgetLabel p, 
    .stMarkdown label, 
    .stMarkdown p:not(button p) {
        color: #111111 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    /* Target the text next to Checkboxes (Power Steering, AC, etc.) */
    .stCheckbox label div[data-testid="stWidgetLabel"] p {
        color: #111111 !important;
    }

    /* Secondary fix for all Markdown-based text in the main area */
    div[data-testid="stMarkdownContainer"] p {
        color: #111111 !important;
    }

    /* Small hack to ensure labels inside columns remain dark */
    [data-testid="column"] .stWidgetLabel p {
        color: #111111 !important;
    }

    /* Buttons - Clean, high-contrast states */
    .stButton > button {
        background-color: var(--primary-text) !important;
        color: #FFFFFF !important; /* Force white text */
        border-radius: 30px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Ensure the text inside the button specifically stays white */
    .stButton > button p, .stButton > button div {
        color: #FFFFFF !important;
    }

    .stButton > button:hover {
        background-color: var(--action-accent) !important;
        color: #FFFFFF !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 92, 26, 0.2);
    }
    
    /* Inputs - Force white background and dark text, clean borders */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input {
        background-color: #ffffff !important;
        border-radius: 8px;
        border: 1px solid #DCDBD8;
        color: #111111 !important;
        caret-color: #111111;
    }
    
    /* Specific targeting for Selectbox to avoid dark corners */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #DCDBD8 !important;
        border-radius: 8px !important;
        color: #111111 !important;
    }
    
    /* Fix for selectbox arrow and text */
    .stSelectbox svg {
        fill: #111111 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: #111111 !important;
    }
    
    /* Checkbox text */
    .stCheckbox label span {
        color: #111111 !important;
    }

</style>
""", unsafe_allow_html=True)


# --- Helpers ---
@st.cache_resource
def load_yolo_model():
    model_path = os.path.join(ROOT_DIR, "damage-detection", "models", "v1.pt")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    return YOLO(model_path)

@st.cache_resource
def load_price_model_artifacts():
    try:
        model_path = os.path.join(ROOT_DIR, "price-model", "best_optimized_model.pkl")
        preproc_path = os.path.join(ROOT_DIR, "price-model", "preprocessing_optimized.pkl")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(preproc_path, 'rb') as f:
            preprocessing = pickle.load(f)
            
        return model, preprocessing
    except Exception as e:
        st.error(f"Error loading price model: {e}")
        return None, None

# --- UI Layout ---

st.title("AutoAnalyze Pro")
st.markdown("### Vehicle Damage & Price Analysis System")

tab1, tab2 = st.tabs(["Damage Detection", "Price Prediction"])

# --- Tab 1: Damage Detection ---
with tab1:
    st.markdown("#### Upload Vehicle Image")
    st.write("Detect dents, scratches, and other exterior damage automatically.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            st.markdown("#### Analysis Results")
            model = load_yolo_model()
            
            if model and st.button("Analyze Damage"):
                with st.spinner('Analyzing...'):
                    # Save temp file for YOLO
                    temp_path = "temp_upload.jpg"
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(temp_path)
                    
                    results = model.predict(temp_path)
                    
                    # Visualize
                    for r in results:
                        im_array = r.plot()  # plot a BGR numpy array of predictions
                        im_rgb = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                        st.image(im_rgb, caption="Detected Damage", use_container_width=True)
                        
                        # Metrics display
                        if len(r.boxes) > 0:
                            st.success(f"Detected {len(r.boxes)} issues.")
                            # Could list specific classes here if mapped
                        else:
                            st.info("No visible damage detected.")
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass

# --- Tab 2: Price Prediction ---
with tab2:
    st.markdown("#### Vehicle Details")
    st.write("Enter vehicle specifications to estimate market value.")
    
    # Load data for dropdowns
    @st.cache_data
    def load_dropdown_data():
        try:
            csv_path = os.path.join(ROOT_DIR, "data", "initial-cleaning", "cleaned_no_outliers.csv")
            return pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Error loading data for dropdowns: {e}")
            return None

    df_options = load_dropdown_data()
    model, preprocessing = load_price_model_artifacts()
    
    if preprocessing and df_options is not None:
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dynamic Dropdowns
            unique_makes = sorted(df_options['Make'].unique())
            make = st.selectbox("Make", unique_makes, index=None, placeholder="Select Make...")
            
            # Filter models based on selected make
            if make:
                filtered_models = sorted(df_options[df_options['Make'] == make]['Model'].unique())
                model_name = st.selectbox("Model", filtered_models, index=None, placeholder="Select Model...")
            else:
                st.selectbox("Model", [], disabled=True, placeholder="Select Make first")
                model_name = None
            
            min_year = int(df_options['YOM'].min())
            max_year = int(df_options['YOM'].max())
            yom = st.number_input("Year of Manufacture", min_value=min_year, max_value=max_year, value=2018)
            
            mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
            
        with col2:
            engine = st.number_input("Engine Capacity (cc)", min_value=600, value=1500)
            
            unique_fuel = sorted(df_options['Fuel Type'].unique())
            fuel_type = st.selectbox("Fuel Type", unique_fuel)
            
            unique_gear = sorted(df_options['Gear'].unique())
            gear = st.selectbox("Gear", unique_gear)
            
            unique_condition = sorted(df_options['Condition'].unique())
            condition = st.selectbox("Condition", unique_condition)
            
            st.markdown("###### Options")
            col_a, col_b = st.columns(2)
            with col_a:
                has_ac = st.checkbox("AC", value=True)
                has_power_steering = st.checkbox("Power Steering", value=True)
            with col_b:
                has_power_mirror = st.checkbox("Power Mirror", value=True)
                has_power_window = st.checkbox("Power Window", value=True)
            
            # Calculate Option Count automatically
            options_count = sum([has_ac, has_power_steering, has_power_mirror, has_power_window])

        st.markdown("<br>", unsafe_allow_html=True) # Add some spacing
        submitted = st.button("Estimate Price") # Changed from form_submit_button to button for interactivity
        
        if submitted:
                if not make or not model_name:
                    st.error("Please select both Make and Model to estimate price.")
                elif model:
                    # Construct input dictionary
                    input_data = {
                        'Make': make,
                        'Model': model_name,
                        'YOM': yom,
                        'Mileage (km)': mileage,
                        'Engine (cc)': engine,
                        'Fuel Type': fuel_type,
                        'Gear': gear,
                        'Condition': condition,
                        'Option_Count': options_count,
                        'Has_AC': 1 if has_ac else 0,
                        'Has_PowerSteering': 1 if has_power_steering else 0,
                        'Has_PowerMirror': 1 if has_power_mirror else 0,
                        'Has_PowerWindow': 1 if has_power_window else 0
                    }
                    
                    prediction, _ = predict_price(input_data, model, preprocessing)
                    
                    if prediction:
                        st.markdown(f"""
                        <div style="background-color: #FF5C1A; color: white; padding: 24px; border-radius: 12px; text-align: center; margin-top: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <h4 style="margin:0; color: white; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">Estimated Market Value</h4>
                            <h1 style="margin: 10px 0; font-size: 3.5em; font-weight: 700; color: white;">LKR {prediction:,.0f}</h1>
                            <p style="margin:0; opacity: 0.9; font-size: 0.9em;">Analysis based on {yom} {make} {model_name} with {mileage:,} km</p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        if not preprocessing:
            st.warning("Price model artifacts not found. Please ensure `price-model/` contains `.pkl` files.")
        if df_options is None:
             st.warning("Data file for dropdowns not found.")
