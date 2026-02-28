import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

def main():
    print("[INFO] 1. Data Preparation")
    data_path = os.path.join(project_root, 'data', 'initial-cleaning', 'cleaned-before_log.csv')
    try:
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()  # Fix trailing spaces in historical data
        print(f"Loaded {len(df)} historical listings")
    except FileNotFoundError:
        print(f"Error: Could not find historical data at {data_path}")
        return

    print("\n[INFO] 2. Feature Engineering")
    features = ['Price', 'Mileage (km)', 'Engine (cc)']
    df_clean = df.dropna(subset=features).copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])
    print("Features (Price, Mileage (km), Engine (cc)) scaled and ready.")

    print("\n[INFO] 3. Training Unsupervised Models")
    # 1. Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    df_clean['Outlier_IF'] = iso_forest.fit_predict(X_scaled)
    print("* Isolation Forest trained")

    # 2. One-Class SVM
    oc_svm = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
    df_clean['Outlier_SVM'] = oc_svm.fit_predict(X_scaled)
    print("* One-Class SVM trained")

    # 3. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    df_clean['Outlier_LOF'] = lof.fit_predict(X_scaled)
    print("* Local Outlier Factor trained")

    print("\n[INFO] 4. Comparing the Results")
    
    # Save full anomaly lists to CSV for review
    if_anomalies = df_clean[df_clean['Outlier_IF'] == -1]
    svm_anomalies = df_clean[df_clean['Outlier_SVM'] == -1]
    lof_anomalies = df_clean[df_clean['Outlier_LOF'] == -1]
    
    if_anomalies.to_csv(os.path.join(project_root, 'data', 'pipeline', 'IF_anomalies.csv'), index=False)
    svm_anomalies.to_csv(os.path.join(project_root, 'data', 'pipeline', 'SVM_anomalies.csv'), index=False)
    lof_anomalies.to_csv(os.path.join(project_root, 'data', 'pipeline', 'LOF_anomalies.csv'), index=False)
    
    print("\n[INFO] Isolation Forest Anomalies (+1 is inlier, -1 is outlier)")
    print(f"Flagged {len(if_anomalies)} rows. Saved full list to data/pipeline/IF_anomalies.csv")
    print(if_anomalies[['Make', 'Model', 'Price', 'Mileage (km)', 'Engine (cc)']].head(10))

    print("\n[INFO] One-Class SVM Anomalies")
    print(f"Flagged {len(svm_anomalies)} rows. Saved full list to data/pipeline/SVM_anomalies.csv")
    print(svm_anomalies[['Make', 'Model', 'Price', 'Mileage (km)', 'Engine (cc)']].head(10))

    print("\n[INFO] LOF Anomalies")
    print(f"Flagged {len(lof_anomalies)} rows. Saved full list to data/pipeline/LOF_anomalies.csv")
    print(lof_anomalies[['Make', 'Model', 'Price', 'Mileage (km)', 'Engine (cc)']].head(10))

    print("\n[INFO] 5. Tricky Synthetic Market Data Tests")
    
    # Test 1: Obvious Garbage
    test1 = [50000000, 50000, 800] # 50M Alto
    
    # Test 2: Tricky High Mileage (Normal price, but 1.5 MILLION kms)
    test2 = [3500000, 1500000, 1000] # e.g. WagonR with insane mileage
    
    # Test 3: Tricky Price (Underpriced drastically)
    test3 = [150000, 20000, 1500] # Brand new sedan for 1.5 Lakhs
    
    fake_rows = pd.DataFrame([test1, test2, test3], columns=['Price', 'Mileage (km)', 'Engine (cc)'])
    fake_scaled = scaler.transform(fake_rows)
    
    iso_preds = iso_forest.predict(fake_scaled)
    svm_preds = oc_svm.predict(fake_scaled)
    
    tests = [
        "Test 1 (50M LKR Alto, 800cc)       ",
        "Test 2 (1.5M km Mileage, 3.5M LKR) ",
        "Test 3 (1.5 Lakhs LKR, Brand new)  "
    ]
    
    print("\n[Results]")
    for i in range(3):
        print(f"{tests[i]}")
        print(f"  -> Isolation Forest : {'CAUGHT' if iso_preds[i] == -1 else 'MISSED'}")
        print(f"  -> One-Class SVM    : {'CAUGHT' if svm_preds[i] == -1 else 'MISSED'}")

    print("\n[INFO] 5. Injecting Synthetic Garbage")
    # Let's create a row that is obviously garbage (a 50 Million LKR Suzuki Alto)
    fake_row = pd.DataFrame([{
        'Price': 50000000, 
        'Mileage (km)': 50000, 
        'Engine (cc)': 800,
        'Make': 'Suzuki',
        'Model': 'Alto'
    }])
    
    # Scale it using the SAME scaler we used for training
    fake_scaled = scaler.transform(fake_row[['Price', 'Mileage (km)', 'Engine (cc)']])
    
    print("Testing Fake Row: 50,000,000 LKR Suzuki Alto (800cc, 50k Mileage)")
    
    # +1 is normal, -1 is anomaly
    iso_pred = iso_forest.predict(fake_scaled)[0]
    svm_pred = oc_svm.predict(fake_scaled)[0]
    
    print(f"Isolation Forest : {'CAUGHT ANOMALY' if iso_pred == -1 else 'MISSED IT (Thought it was normal)'}")
    print(f"One-Class SVM    : {'CAUGHT ANOMALY' if svm_pred == -1 else 'MISSED IT (Thought it was normal)'}")
    
    print("\nNote: Local Outlier Factor (LOF) typically cannot predict on new unseen rows after fitting,")
    print("it is mostly used for analyzing existing static datasets, so we rely on IF and SVM for pipelines.")

if __name__ == "__main__":
    main()
