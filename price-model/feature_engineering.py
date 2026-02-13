"""
Feature engineering utilities
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    AVG_KM_PER_YEAR, LOW_MILEAGE_THRESHOLD, MEDIUM_MILEAGE_MAX,
    HIGH_MILEAGE_THRESHOLD, VERY_HIGH_MILEAGE_THRESHOLD,
    SMALL_ENGINE_MAX, MEDIUM_ENGINE_MAX, LARGE_ENGINE_MIN, VERY_LARGE_ENGINE_MIN,
    PREMIUM_BRAND_QUANTILE, LUXURY_BRAND_QUANTILE, RECENT_MODEL_YEAR,
    CONDITION_MAP, FUEL_TYPE_FREQ, GEAR_FREQ
)


def calculate_all_features(input_data, brand_stats, model_stats):
    """
    Calculate all engineered features from input data
    """
    df = pd.DataFrame([input_data])

    # Get current year for age calculation
    current_year = datetime.now().year
    vehicle_age = current_year - df['YOM'].values[0]
    df['Vehicle Age'] = vehicle_age

    # Calculate brand features
    df = _calculate_brand_features(df, brand_stats)

    # Calculate model features
    df = _calculate_model_features(df, model_stats)

    # Calculate brand-model combination features
    df = _calculate_brand_model_features(df, brand_stats, model_stats)

    # Calculate age-based features
    df = _calculate_age_features(df, vehicle_age)

    # Calculate mileage features
    df = _calculate_mileage_features(df, vehicle_age)

    # Calculate engine features
    df = _calculate_engine_features(df, vehicle_age)

    # Calculate interaction features
    df = _calculate_interaction_features(df, vehicle_age)

    # Calculate ratio features
    df = _calculate_ratio_features(df, vehicle_age)

    # Calculate categorical encodings
    df = _calculate_categorical_encodings(df)

    return df


def _calculate_brand_features(df, brand_stats):
    """Calculate brand-related features"""
    make = df['Make'].values[0]

    if make in brand_stats.index:
        df['Brand_Avg_Price'] = brand_stats.loc[make, 'mean']
        df['Brand_Median_Price'] = brand_stats.loc[make, 'median']
        df['Brand_Price_Std'] = brand_stats.loc[make, 'std'] if not pd.isna(brand_stats.loc[make, 'std']) else 0
        df['Brand_Popularity'] = brand_stats.loc[make, 'count']
        df['Brand_Min_Price'] = brand_stats.loc[make, 'min']
        df['Brand_Max_Price'] = brand_stats.loc[make, 'max']
    else:
        # Use overall average if brand not found
        df['Brand_Avg_Price'] = brand_stats['mean'].mean()
        df['Brand_Median_Price'] = brand_stats['median'].mean()
        df['Brand_Price_Std'] = brand_stats['std'].mean()
        df['Brand_Popularity'] = brand_stats['count'].mean()
        df['Brand_Min_Price'] = brand_stats['min'].mean()
        df['Brand_Max_Price'] = brand_stats['max'].mean()

    df['Brand_Price_Range'] = df['Brand_Max_Price'] - df['Brand_Min_Price']

    # Premium indicators
    overall_avg = brand_stats['mean'].mean()
    premium_threshold = brand_stats['mean'].quantile(PREMIUM_BRAND_QUANTILE)
    luxury_threshold = brand_stats['mean'].quantile(LUXURY_BRAND_QUANTILE)

    df['Is_Premium_Brand'] = 1 if df['Brand_Avg_Price'].values[0] > premium_threshold else 0
    df['Is_Luxury_Brand'] = 1 if df['Brand_Avg_Price'].values[0] > luxury_threshold else 0
    df['Brand_Prestige_Score'] = df['Brand_Avg_Price'] / overall_avg

    return df


def _calculate_model_features(df, model_stats):
    """Calculate model-related features"""
    model_name = df['Model'].values[0]

    if model_name in model_stats.index:
        df['Model_Avg_Price'] = model_stats.loc[model_name, 'mean']
        df['Model_Median_Price'] = model_stats.loc[model_name, 'median']
        df['Model_Price_Std'] = model_stats.loc[model_name, 'std'] if not pd.isna(
            model_stats.loc[model_name, 'std']) else 0
        df['Model_Popularity'] = model_stats.loc[model_name, 'count']
    else:
        df['Model_Avg_Price'] = model_stats['mean'].mean()
        df['Model_Median_Price'] = model_stats['median'].mean()
        df['Model_Price_Std'] = model_stats['std'].mean()
        df['Model_Popularity'] = model_stats['count'].mean()

    # Model prestige
    overall_avg = model_stats['mean'].mean()
    df['Model_Prestige_Score'] = df['Model_Avg_Price'] / overall_avg

    return df


def _calculate_brand_model_features(df, brand_stats, model_stats):

    return df


def _calculate_age_features(df, vehicle_age):

    return df


def _calculate_mileage_features(df, vehicle_age):

    return df


def _calculate_engine_features(df, vehicle_age):

    return df


def _calculate_interaction_features(df, vehicle_age):

    return df


def _calculate_ratio_features(df, vehicle_age):


    return df


def _calculate_categorical_encodings(df):


    return df