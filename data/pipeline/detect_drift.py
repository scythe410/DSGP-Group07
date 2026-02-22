
import pandas as pd
import numpy as np
import os
import logging
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REFERENCE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "initial-cleaning", "cleaned-before_log.csv")
NEW_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_listings_latest.csv")

# Drift Thresholds
PRICE_MEAN_DRIFT_THRESHOLD = 0.05  # 5% change
DISTRIBUTION_P_VALUE_THRESHOLD = 0.05 # P-value for Kolmogorov-Smirnov test

def load_data():
    if not os.path.exists(REFERENCE_DATA_PATH):
        logging.error(f"Reference data not found at {REFERENCE_DATA_PATH}")
        return None, None
        
    if not os.path.exists(NEW_DATA_PATH):
        logging.error(f"New data not found at {NEW_DATA_PATH}")
        return None, None
        
    try:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        new_df = pd.read_csv(NEW_DATA_PATH)
        
        # Standardize columns
        ref_df.columns = [c.strip() for c in ref_df.columns]
        new_df.columns = [c.strip() for c in new_df.columns]
        
        return ref_df, new_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None

def check_price_drift(ref_df, new_df):
    """Check for drift in Price column"""
    if 'Price' not in ref_df.columns or 'Price' not in new_df.columns:
        logging.warning("Price column missing. Cannot check price drift.")
        return False
        
    ref_mean = ref_df['Price'].mean()
    new_mean = new_df['Price'].mean()
    
    drift = abs(new_mean - ref_mean) / ref_mean
    logging.info(f"Price Mean Drift: {drift:.4f} (Ref: {ref_mean:.2f}, New: {new_mean:.2f})")
    
    if drift > PRICE_MEAN_DRIFT_THRESHOLD:
        logging.warning(f"Significant Price Mean Drift detected (> {PRICE_MEAN_DRIFT_THRESHOLD})")
        return True
        
    # Statistical test (KS test) for distribution change
    # Null hypothesis: The two distributions are identical
    # If p-value < threshold, we reject null hypothesis -> Drift detected
    try:
        statistic, p_value = stats.ks_2samp(ref_df['Price'].dropna(), new_df['Price'].dropna())
        logging.info(f"Price Distribution KS Test: Statistic={statistic:.4f}, P-value={p_value:.4f}")
        
        if p_value < DISTRIBUTION_P_VALUE_THRESHOLD:
             logging.warning("Significant Price Distribution Drift detected (KS Test)")
             return True
    except Exception as e:
        logging.warning(f"Could not perform KS test: {e}")
        
    return False

def check_categorical_drift(ref_df, new_df, col):
    """Check for drift in categorical columns"""
    if col not in ref_df.columns or col not in new_df.columns:
        return False
        
    ref_dist = ref_df[col].value_counts(normalize=True)
    new_dist = new_df[col].value_counts(normalize=True)
    
    # Calculate simple variation distance or similar
    # align indices and sum absolute differences
    all_cats = set(ref_dist.index) | set(new_dist.index)
    diff_sum = 0
    for cat in all_cats:
        ref_val = ref_dist.get(cat, 0)
        new_val = new_dist.get(cat, 0)
        diff_sum += abs(ref_val - new_val)
        
    logging.info(f"Categorical Drift for {col}: {diff_sum:.4f}")
    
    if diff_sum > 0.1: # 10% total variation
        logging.warning(f"Significant Categorical Drift in {col}")
        return True
        
    return False

def main():
    logging.info("Starting Drift Detection...")
    ref_df, new_df = load_data()
    
    if ref_df is None or new_df is None:
        logging.error("Could not load data. Exiting.")
        return
        
    drift_detected = False
    
    if check_price_drift(ref_df, new_df):
        drift_detected = True
        
    if check_categorical_drift(ref_df, new_df, 'Make'):
        drift_detected = True
        
    if drift_detected:
        logging.warning("DRIFT DETECTED! Triggering Retraining...")
        # Create a trigger file or return appropriate exit code
        with open(os.path.join(PROCESSED_DATA_DIR, "drift_detected.flag"), "w") as f:
            f.write("True")
            
        print("DRIFT_DETECTED=True") # For shell script parsing
    else:
        logging.info("No significant drift detected.")
        if os.path.exists(os.path.join(PROCESSED_DATA_DIR, "drift_detected.flag")):
            os.remove(os.path.join(PROCESSED_DATA_DIR, "drift_detected.flag"))
        print("DRIFT_DETECTED=False")

if __name__ == "__main__":
    main()
