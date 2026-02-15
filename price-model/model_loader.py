"""
Model loading utilities
"""

import pickle
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import MODEL_PATH, PREPROCESSING_PATH


@st.cache_resource
def load_model_and_preprocessing():
    """
    Load the trained model and preprocessing objects
    """
    try:
        # Try to load from current directory
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        with open(PREPROCESSING_PATH, 'rb') as f:
            preprocessing = pickle.load(f)

        return model, preprocessing

    except FileNotFoundError:
        # Try to load from parent directory
        try:
            parent_path = Path(__file__).parent.parent.parent
            model_path = parent_path / MODEL_PATH
            preprocessing_path = parent_path / PREPROCESSING_PATH

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            with open(preprocessing_path, 'rb') as f:
                preprocessing = pickle.load(f)

            return model, preprocessing

        except Exception as e:
            st.error(f"Error loading model files: {str(e)}")
            st.info(f"Please ensure '{MODEL_PATH}' and '{PREPROCESSING_PATH}' are in the correct directory.")
            st.stop()

    except Exception as e:
        st.error(f"Unexpected error loading model: {str(e)}")
        st.stop()


def get_available_makes(preprocessing):
    """
    Get list of available car makes from preprocessing data
    """
    import pandas as pd
    brand_stats = pd.DataFrame(preprocessing['brand_stats'])
    return sorted(brand_stats.index.tolist())


def get_available_models(preprocessing):
    """
    Get list of available car models from preprocessing data
    """
    import pandas as pd
    model_stats = pd.DataFrame(preprocessing['model_stats'])
    return sorted(model_stats.index.tolist())