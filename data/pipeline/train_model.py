
import pandas as pd
import numpy as np
import os
import sys
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRICE_MODEL_DIR = os.path.join(PROJECT_ROOT, "price-model")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
REFERENCE_DATA_PATH = os.path.join(DATA_DIR, "initial-cleaning", "cleaned-before_log.csv")
NEW_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_listings_latest.csv")

# Add price-model to path for imports
sys.path.append(PRICE_MODEL_DIR)

try:
    from feature_engineering import calculate_all_features
    import config
except ImportError:
    logging.error("Could not import feature_engineering or config from price-model. Check paths.")
    sys.exit(1)

def load_combined_data():
    """Load only new data for training (as old data prices are outdated)"""
    if os.path.exists(NEW_DATA_PATH):
        try:
            new_df = pd.read_csv(NEW_DATA_PATH)
            new_df.columns = [c.strip() for c in new_df.columns]
            logging.info(f"Loaded new data for training: {len(new_df)} rows")
            return new_df
        except Exception as e:
            logging.error(f"Error loading new data: {e}")
            return None
    else:
        logging.error("No new data found for training.")
        return None

def compute_stats(df):
    """Compute brand and model statistics"""
    # Brand stats
    brand_stats = df.groupby('Make')['Price'].agg(['mean', 'median', 'std', 'count', 'min', 'max'])
    
    # Model stats
    model_stats = df.groupby('Model')['Price'].agg(['mean', 'median', 'std', 'count'])
    
    return brand_stats, model_stats

def prepare_features(df, brand_stats, model_stats):
    """Generate features for the entire dataframe"""
    logging.info("Generating features...")
    feature_rows = []
    
    for _, row in df.iterrows():
        input_data = row.to_dict()
        try:
            feat_df = calculate_all_features(input_data, brand_stats, model_stats)
            feature_rows.append(feat_df)
        except Exception as e:
            # logging.warning(f"Error generating features for row: {e}")
            continue
            
    if not feature_rows:
        return None
        
    features_df = pd.concat(feature_rows, ignore_index=True)
    return features_df

def train_model(X, y):
    """Train the model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info(f"Training on {len(X_train)} samples...")
    
    # Try XGBoost, fallback to RandomForest
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=42)
        model_name = "XGBoost"
    except ImportError:
        logging.info("XGBoost not found, using RandomForest")
        model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
        model_name = "RandomForest"
        
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f"{model_name} Performance: MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
    
    model.fit(X, y)
    return model, r2, mae

def main():
    logging.info("Starting Model Retraining...")
    
    # 1. Load Data
    df = load_combined_data()
    if df is None or len(df) < 10:
        logging.error("Not enough data to train.")
        return
    
    # 2. Compute Stats (for feature engineering and saving)
    brand_stats, model_stats = compute_stats(df)
    
    # 3. Feature Engineering
    features_df = prepare_features(df, brand_stats, model_stats)
    
    if features_df is None:
        logging.error("Feature engineering failed.")
        return
        
    # 4. Prepare X, y
    # Identify categorical columns to encode
    cat_cols = ['Make', 'Model', 'Gear', 'Fuel Type', 'Condition']
    label_encoders = {}
    
    for col in cat_cols:
        if col in features_df.columns:
            le = LabelEncoder()
            features_df[col] = features_df[col].astype(str)
            features_df[col] = le.fit_transform(features_df[col])
            label_encoders[col] = le
            
    exclude_cols = ['Price', 'Url', 'Title', 'Description', 'Contact', 'Location']
    
    feature_columns = [c for c in features_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(features_df[c])]
    
    X = features_df[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['Price'] # Original prices
    
    # 5. Train
    model, r2, mae = train_model(X, y)
    
    if r2 < 0.5: # Arbitrary threshold
        logging.warning(f"Model R2 score {r2:.4f} is low. Creating backup but maybe not deploying automatically.")
        
    # 6. Save Artifacts
    # Construct preprocessing dict
    preprocessing = {
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'brand_stats': brand_stats.to_dict(),
        'model_stats': model_stats.to_dict()
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Model
    model_path = os.path.join(PRICE_MODEL_DIR, f"model_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Saved model to {model_path}")
    
    # Save Preprocessing
    prep_path = os.path.join(PRICE_MODEL_DIR, f"preprocessing_{timestamp}.pkl")
    with open(prep_path, 'wb') as f:
        pickle.dump(preprocessing, f)
    logging.info(f"Saved preprocessing to {prep_path}")
    
    # Update 'latest' retrained model
    if r2 > 0.5:
        retrained_model_path = os.path.join(PRICE_MODEL_DIR, "retrained_model.pkl")
        retrained_prep_path  = os.path.join(PRICE_MODEL_DIR, "preprocessing_retrained.pkl")

        with open(retrained_model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(retrained_prep_path, 'wb') as f:
            pickle.dump(preprocessing, f)

        logging.info(f"Saved retrained model to {retrained_model_path}")

        # ── Push to HuggingFace Hub ──────────────────────────────────────────
        hf_token = os.environ.get("HF_TOKEN")
        hf_pushed = False
        if hf_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                HF_REPO = "scythe410/sri-lankan-vehicle-price-prediction"

                # Push model and preprocessing with a timestamped commit message
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
                api.upload_file(
                    path_or_fileobj=retrained_model_path,
                    path_in_repo="best_optimized_model.pkl",
                    repo_id=HF_REPO,
                    repo_type="model",
                    commit_message=f"Auto-retrain: R\u00b2={r2:.4f}, MAE={mae:.0f} LKR ({timestamp_str})",
                )
                api.upload_file(
                    path_or_fileobj=retrained_prep_path,
                    path_in_repo="preprocessing_optimized.pkl",
                    repo_id=HF_REPO,
                    repo_type="model",
                    commit_message=f"Auto-retrain preprocessing ({timestamp_str})",
                )
                logging.info(f"[HF] Pushed retrained model to {HF_REPO}")
                hf_pushed = True

                # Write success flag so the CI summary can detect it
                flag_path = os.path.join(PRICE_MODEL_DIR, ".hf_push_success")
                with open(flag_path, "w") as f:
                    f.write(f"Pushed at {timestamp_str} | R²={r2:.4f} MAE={mae:.0f}\n")

                # ── Restart HuggingFace Space so it loads the new model ──────────
                HF_SPACE = "scythe410/dsgp_007"  # underscore, not hyphen
                try:
                    import requests
                    restart_url = f"https://huggingface.co/api/spaces/{HF_SPACE}/restart"
                    resp = requests.post(restart_url,
                                         headers={"Authorization": f"Bearer {hf_token}"},
                                         timeout=30)
                    if resp.status_code == 200:
                        logging.info(f"[HF] Space '{HF_SPACE}' restart triggered \u2014 new model will load on startup.")
                    else:
                        logging.warning(f"[HF] Space restart returned HTTP {resp.status_code}: {resp.text[:200]}")
                except Exception as e:
                    logging.warning(f"[HF] Could not restart Space (non-fatal): {e}")

            except Exception as e:
                logging.error(f"[HF] Upload failed: {e}. Retrained model is saved locally only.")
        else:
            logging.warning("HF_TOKEN not set \u2014 skipping HuggingFace push. Model saved locally only.")

        return hf_pushed
    else:
        logging.warning(f"R\u00b2 = {r2:.4f} is below 0.5 \u2014 model NOT pushed to HuggingFace.")
        return False

if __name__ == "__main__":
    main()
