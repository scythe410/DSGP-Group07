import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

def detect_anomalies(raw_data_path, output_clean_path, output_quarantine_path, historical_data_path):
    print("[INFO] Running Anomaly Detection (One-Class SVM)")
    
    # 1. Load historical data to fit the Scaler & Model
    print("Loading historical baseline...")
    try:
        df_hist = pd.read_csv(historical_data_path)
        df_hist.columns = df_hist.columns.str.strip()
    except FileNotFoundError:
        print(f"ERROR: Historical data not found at {historical_data_path}. Cannot fit Anomaly Model.")
        return False
        
    features = ['Price', 'Mileage (km)', 'Engine (cc)']
    df_hist_clean = df_hist.dropna(subset=features).copy()
    
    scaler = StandardScaler()
    X_hist_scaled = scaler.fit_transform(df_hist_clean[features])
    
    oc_svm = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
    oc_svm.fit(X_hist_scaled)
    print("* Model fitted on historical baseline")
    
    # 2. Load today's scraped data
    print(f"Loading today's scrape: {raw_data_path}")
    try:
        df_today = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"ERROR: Scrape file not found at {raw_data_path}")
        return False
        
    if df_today.empty:
        print("Warning: Scrape file is empty. Skipping anomaly detection.")
        return True
        
    # We must ensure today's scrape has the correct column names for scaling
    # Assuming scrape_listings.py outputs similar columns. If not, mapping is required here.
    # We will safely drop na for those specific columns just to predict, but keep the original rows for quarantine
    df_pred_set = df_today.dropna(subset=features).copy()
    
    if df_pred_set.empty:
        print("Warning: No valid rows with Price, Mileage, and Engine capacity found to test.")
        df_today.to_csv(output_clean_path, index=False)
        return True
        
    # Scale today's data using the fitted scaler
    X_today_scaled = scaler.transform(df_pred_set[features])
    
    # Predict: +1 represents inlier, -1 represents outlier/anomaly
    predictions = oc_svm.predict(X_today_scaled)
    df_pred_set['Anomaly_Flag'] = predictions
    
    # Merge the flags back to the full dataset. Rows missing features get NaN, we'll keep them as "normal" or handle in cleaner
    df_today = df_today.merge(df_pred_set[['Anomaly_Flag']], left_index=True, right_index=True, how='left')
    df_today['Anomaly_Flag'] = df_today['Anomaly_Flag'].fillna(1) # default to normal if missing features
    
    # Split the dataset
    df_clean = df_today[df_today['Anomaly_Flag'] == 1].drop(columns=['Anomaly_Flag'])
    df_quarantine = df_today[df_today['Anomaly_Flag'] == -1].drop(columns=['Anomaly_Flag'])
    
    print(f"Total Rows: {len(df_today)}")
    print(f"Valid (Passed Anomaly Check): {len(df_clean)}")
    print(f"Quarantined Anomalies: {len(df_quarantine)}")
    
    # Save outputs
    df_clean.to_csv(output_clean_path, index=False)
    
    if not df_quarantine.empty:
        print(f"WARNING: Found {len(df_quarantine)} anomalies! Saved to quarantine log.")
        df_quarantine.to_csv(output_quarantine_path, index=False)
        
    print("Anomaly Detection complete.")
    return True

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # For testing the script standalone
    raw_path = os.path.join(project_root, 'data', 'raw', 'listings_latest.csv')
    clean_path = os.path.join(project_root, 'data', 'raw', 'listings_anomaly_checked.csv')
    quarantine_path = os.path.join(project_root, 'data', 'raw', 'quarantined_ads.csv')
    historical_path = os.path.join(project_root, 'data', 'initial-cleaning', 'cleaned-before_log.csv')
    
    # Create dummy raw data if testing directly
    if not os.path.exists(raw_path):
        print("Creating dummy raw data for standalone test...")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        pd.DataFrame([{
            'Price': 50000000, 'Mileage (km)': 50000, 'Engine (cc)': 800, 'Make': 'Suzuki', 'Model': 'Alto'
        }]).to_csv(raw_path, index=False)
    
    detect_anomalies(raw_path, clean_path, quarantine_path, historical_path)
