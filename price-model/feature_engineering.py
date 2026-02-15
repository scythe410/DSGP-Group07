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
    """Calculate brand-model combination features"""

    df['Brand_Model_Avg_Price'] = (df['Brand_Avg_Price'] + df['Model_Avg_Price']) / 2
    df['Brand_Model_Median_Price'] = (df['Brand_Median_Price'] + df['Model_Median_Price']) / 2
    df['Brand_Model_Count'] = (df['Brand_Popularity'] + df['Model_Popularity']) / 2

    return df


def _calculate_age_features(df, vehicle_age):
    """Calculate age-related features"""
    # Polynomial transformations
    df['Age_Squared'] = vehicle_age ** 2
    df['Age_Cubed'] = vehicle_age ** 3
    df['Age_Fourth'] = vehicle_age ** 4

    # Non-linear transformations
    df['Age_Log'] = np.log1p(vehicle_age)
    df['Age_Sqrt'] = np.sqrt(vehicle_age)
    df['Age_Inv'] = 1 / (vehicle_age + 1)
    df['Age_Exp'] = np.exp(-vehicle_age / 10)

    # Categorical age features
    df['YOM_Decade'] = (df['YOM'] // 10) * 10
    df['Years_Since_2000'] = df['YOM'] - 2000
    df['Is_Brand_New'] = 1 if vehicle_age == 0 else 0
    df['Is_New'] = 1 if vehicle_age <= 2 else 0
    df['Is_Recent'] = 1 if vehicle_age <= 5 else 0
    df['Is_Old'] = 1 if vehicle_age > 10 else 0
    df['Is_Vintage'] = 1 if vehicle_age > 20 else 0
    df['Is_Recent_Model'] = 1 if df['YOM'].values[0] >= RECENT_MODEL_YEAR else 0

    return df


def _calculate_mileage_features(df, vehicle_age):
    """Calculate mileage-related features"""
    mileage = df['Mileage (km)'].values[0]

    # Usage intensity
    df['Km_per_Year'] = mileage / (vehicle_age + 1)

    # Transformations
    df['Mileage_Log'] = np.log1p(mileage)
    df['Mileage_Sqrt'] = np.sqrt(mileage)
    df['Mileage_Squared'] = mileage ** 2

    # Expected vs actual
    df['Expected_Mileage'] = AVG_KM_PER_YEAR * vehicle_age
    df['Mileage_vs_Expected'] = mileage - df['Expected_Mileage']
    df['Mileage_Ratio_Expected'] = mileage / (df['Expected_Mileage'] + 1)

    # Categorical mileage features
    df['Is_Low_Mileage'] = 1 if mileage < LOW_MILEAGE_THRESHOLD else 0
    df['Is_Medium_Mileage'] = 1 if LOW_MILEAGE_THRESHOLD <= mileage < MEDIUM_MILEAGE_MAX else 0
    df['Is_High_Mileage'] = 1 if mileage >= HIGH_MILEAGE_THRESHOLD else 0
    df['Is_Very_High_Mileage'] = 1 if mileage > VERY_HIGH_MILEAGE_THRESHOLD else 0
    df['Is_Below_Avg_Usage'] = 1 if df['Km_per_Year'].values[0] < AVG_KM_PER_YEAR else 0
    df['Is_Heavy_Usage'] = 1 if df['Km_per_Year'].values[0] > 20000 else 0

    return df


def _calculate_engine_features(df, vehicle_age):
    """Calculate engine-related features"""
    engine = df['Engine (cc)'].values[0]

    # Transformations
    df['Engine_Log'] = np.log1p(engine)
    df['Engine_Sqrt'] = np.sqrt(engine)
    df['Engine_Squared'] = engine ** 2

    # Categorical engine features
    df['Is_Small_Engine'] = 1 if engine <= SMALL_ENGINE_MAX else 0
    df['Is_Medium_Engine'] = 1 if SMALL_ENGINE_MAX < engine <= MEDIUM_ENGINE_MAX else 0
    df['Is_Large_Engine'] = 1 if engine > LARGE_ENGINE_MIN else 0
    df['Is_Very_Large_Engine'] = 1 if engine > VERY_LARGE_ENGINE_MIN else 0

    # Relative features
    df['Engine_per_Year'] = engine / (vehicle_age + 1)
    mileage = df['Mileage (km)'].values[0]
    df['Engine_per_1000km'] = engine / (mileage / 1000 + 1)

    return df


def _calculate_interaction_features(df, vehicle_age):
    """Calculate interaction features between variables"""
    mileage = df['Mileage (km)'].values[0]
    engine = df['Engine (cc)'].values[0]

    # Age interactions
    df['Age_x_Mileage'] = vehicle_age * mileage
    df['Age_x_Engine'] = vehicle_age * engine
    df['Age_x_KmPerYear'] = vehicle_age * df['Km_per_Year']
    df['Age_x_Brand_Avg'] = vehicle_age * df['Brand_Avg_Price']
    df['Age_x_Model_Avg'] = vehicle_age * df['Model_Avg_Price']

    # Premium interactions
    df['Premium_x_Age'] = df['Is_Premium_Brand'] * vehicle_age
    df['Luxury_x_Age'] = df['Is_Luxury_Brand'] * vehicle_age
    df['Premium_x_Mileage'] = df['Is_Premium_Brand'] * mileage

    # Other interactions
    df['Engine_x_Mileage'] = engine * mileage
    df['Options_x_Age'] = df['Option_Count'] * vehicle_age
    df['Options_x_Brand_Avg'] = df['Option_Count'] * df['Brand_Avg_Price']
    df['KmPerYear_x_Engine'] = df['Km_per_Year'] * engine
    df['Brand_Prestige_x_Age'] = df['Brand_Prestige_Score'] * vehicle_age

    return df


def _calculate_ratio_features(df, vehicle_age):
    """Calculate ratio features"""
    engine = df['Engine (cc)'].values[0]

    df['Options_Ratio'] = df['Option_Count'] / 4
    df['Price_to_Engine_Ratio'] = df['Brand_Avg_Price'] / (engine + 1)
    df['Depreciation_Rate'] = df['Brand_Avg_Price'] / (vehicle_age + 1)

    return df


def _calculate_categorical_encodings(df):
    """Calculate categorical encoding features"""
    # Frequency encodings
    df['Fuel_Type_Freq'] = FUEL_TYPE_FREQ.get(df['Fuel Type'].values[0], 2000)
    df['Gear_Freq'] = GEAR_FREQ.get(df['Gear'].values[0], 2500)

    # Ordinal encoding
    df['Condition_Ordinal'] = CONDITION_MAP.get(df['Condition'].values[0], 2)

    # Binary indicators
    df['Is_Petrol'] = 1 if df['Fuel Type'].values[0] == 'Petrol' else 0
    df['Is_Diesel'] = 1 if df['Fuel Type'].values[0] == 'Diesel' else 0
    df['Is_Hybrid'] = 1 if df['Fuel Type'].values[0] == 'Hybrid' else 0

    df['Is_Auto'] = 1 if df['Gear'].values[0] == 'Auto' else 0
    df['Is_Manual'] = 1 if df['Gear'].values[0] == 'Manual' else 0

    return df