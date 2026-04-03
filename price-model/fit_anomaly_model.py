"""
One-time script: Fits the OneClassSVM anomaly detector on the historical
baseline data and saves it as anomaly_model.pkl alongside the price model.

Run this once, then upload anomaly_model.pkl to HuggingFace Hub.
"""

import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_CSV = os.path.join(ROOT_DIR, "data", "initial-cleaning", "cleaned-before_log.csv")
OUTPUT_PKL    = os.path.join(ROOT_DIR, "price-model", "anomaly_model.pkl")

# Features used for anomaly detection
# These match what detect_anomalies.py uses in the pipeline
FEATURES = ["Price", "Mileage (km)", "Engine (cc)"]

# Bounds used as a fast, interpretable hard-filter BEFORE the SVM.
# Values outside these ranges are immediately anomalous regardless of SVM.
HARD_BOUNDS = {
    "Engine (cc)":  (600,    6000),    # no real consumer car has < 600cc or > 6000cc
    "Mileage (km)": (0,      500_000), # negative mileage or > 500k is implausible
    "YOM":          (1990,   2026),    # year of manufacture sanity range
    "Price":        (100_000, 250_000_000),  # LKR: below 100k or above 250M is spam
}

def main():
    print(f"[INFO] Loading reference data from {REFERENCE_CSV}")
    df = pd.read_csv(REFERENCE_CSV)
    df.columns = [c.strip() for c in df.columns]

    df_clean = df.dropna(subset=FEATURES).copy()
    print(f"[INFO] Training on {len(df_clean):,} rows")

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[FEATURES])

    # Fit OneClassSVM
    # nu=0.01 means we expect ~1% of training data to be outliers
    oc_svm = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
    oc_svm.fit(X_scaled)

    # Quick self-evaluation
    preds = oc_svm.predict(X_scaled)
    inlier_pct = (preds == 1).mean() * 100
    print(f"[INFO] Inlier rate on training data: {inlier_pct:.1f}%")
    print(f"[INFO] (nu=0.01 → expect ~99.0%)")

    # Save bundle
    bundle = {
        "scaler":      scaler,
        "model":       oc_svm,
        "features":    FEATURES,
        "hard_bounds": HARD_BOUNDS,
    }

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n[✓] Saved anomaly model bundle to: {OUTPUT_PKL}")
    print("     Bundle keys: scaler, model, features, hard_bounds")


if __name__ == "__main__":
    main()
