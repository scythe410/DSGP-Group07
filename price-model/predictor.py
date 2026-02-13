"""
Prediction utilities
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_engineering import calculate_all_features


def predict_price(input_features, model, preprocessing):
    """
    Make prediction using the loaded model
    """
    try:
        import pandas as pd

        # Extract preprocessing objects
        label_encoders = preprocessing['label_encoders']
        feature_columns = preprocessing['feature_columns']
        brand_stats = pd.DataFrame(preprocessing['brand_stats'])
        model_stats = pd.DataFrame(preprocessing['model_stats'])

        # Calculate all engineered features
        df_features = calculate_all_features(input_features, brand_stats, model_stats)

        # Encode categorical variables
        for col in ['Make', 'Model', 'Gear', 'Fuel Type', 'Condition']:
            if col in label_encoders:
                try:
                    df_features[col] = label_encoders[col].transform([df_features[col].values[0]])[0]
                except:
                    # If value not seen during training, use most common (0)
                    df_features[col] = 0

        # Select only the features used by the model
        X = df_features[feature_columns].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # Make prediction
        prediction = model.predict(X)[0]

        return prediction, df_features

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None


def calculate_price_range(predicted_price, percentage=0.10):
    """
    Calculate price range based on prediction
    """
    min_price = predicted_price * (1 - percentage)
    max_price = predicted_price * (1 + percentage)
    return min_price, max_price


def calculate_mileage_score(mileage, max_mileage=200000):
    """
    Calculate mileage score (0-100)
    """
    score = 100 - min((mileage / max_mileage) * 100, 100)
    return max(0, score)


def calculate_age_score(vehicle_age, max_age=25):
    """
    Calculate age score (0-100)
    """
    score = 100 - min((vehicle_age / max_age) * 100, 100)
    return max(0, score)


def calculate_depreciation(brand_avg_price, predicted_price):
    """
    Calculate depreciation percentage
    """
    if brand_avg_price > 0:
        depreciation = ((brand_avg_price - predicted_price) / brand_avg_price * 100)
        return max(0, depreciation)
    return 0


def generate_depreciation_curve(brand_avg_price, vehicle_age, num_years=5):
    """
    Generate depreciation curve data
    """
    years = list(range(0, vehicle_age + num_years))

    # Depreciation rates: 15% first year, 10% next 2 years, 8% after
    prices = []
    for y in years:
        depreciation = 0.15 * min(y, 1) + 0.10 * max(0, min(y - 1, 2)) + 0.08 * max(0, y - 3)
        price = brand_avg_price * (1 - depreciation)
        prices.append(max(0, price))

    return years, prices