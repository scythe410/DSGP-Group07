"""
Configuration settings for the Car Price Prediction app
"""

# File paths
MODEL_PATH = 'best_optimized_model.pkl'
PREPROCESSING_PATH = 'preprocessing_optimized.pkl'

# Application settings
APP_TITLE = "Car Price Predictor"
APP_ICON = "Car"
PAGE_LAYOUT = "wide"

# Available options for dropdowns
AVAILABLE_GEARS = ['Automatic', 'Manual', 'CVT']
AVAILABLE_FUEL_TYPES = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
AVAILABLE_CONDITIONS = ['Brand New', 'Unregistered', 'Used', 'Reconditioned']

# Input ranges
MIN_YEAR = 1990
MIN_MILEAGE = 0
MAX_MILEAGE = 500000
MIN_ENGINE = 500
MAX_ENGINE = 6000

# Default values
DEFAULT_YEAR = 2020
DEFAULT_MILEAGE = 50000
DEFAULT_ENGINE = 1500

# Mileage thresholds
LOW_MILEAGE_THRESHOLD = 30000
MEDIUM_MILEAGE_MAX = 100000
HIGH_MILEAGE_THRESHOLD = 100000
VERY_HIGH_MILEAGE_THRESHOLD = 200000

# Average usage
AVG_KM_PER_YEAR = 15000

# Engine thresholds
SMALL_ENGINE_MAX = 1000
MEDIUM_ENGINE_MAX = 1500
LARGE_ENGINE_MIN = 2000
VERY_LARGE_ENGINE_MIN = 3000

# Brand thresholds
PREMIUM_BRAND_QUANTILE = 0.75
LUXURY_BRAND_QUANTILE = 0.90

# UI Colors
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#2c3e50"
GRADIENT_START = "#667eea"
GRADIENT_END = "#764ba2"

# Condition mapping
CONDITION_MAP = {
    'Unregistered': 4,
    'Brand New': 3,
    'Used': 2,
    'Reconditioned': 1
}

# Fuel type frequency (approximate)
FUEL_TYPE_FREQ = {
    'Petrol': 5000,
    'Diesel': 3000,
    'Hybrid': 1000,
    'Electric': 500
}

# Gear frequency (approximate)
GEAR_FREQ = {
    'Automatic': 4000,
    'Manual': 3500,
    'CVT': 1500
}

# Scoring thresholds
MILEAGE_SCORE_MAX = 200000
AGE_SCORE_MAX = 25

# Recent model year threshold
RECENT_MODEL_YEAR = 2020

# Prediction confidence
PRICE_RANGE_PERCENTAGE = 0.10  # ±10%