
import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_raw_data():
    """Load all CSV files from raw directory"""
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not all_files:
        logging.warning("No raw data files found.")
        return None
    
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            logging.error(f"Error reading {filename}: {e}")
            
    if not df_list:
        return None
        
    return pd.concat(df_list, ignore_index=True)

def clean_data(df):
    """Apply cleaning transformations based on user's manual logic"""
    logging.info(f"Initial shape: {df.shape}")
    
    # 0. Initial Column Cleanup
    df.columns = [c.strip() for c in df.columns]
    
    # 1. Pre-processing types
    if 'YOM' in df.columns:
        df['YOM'] = pd.to_numeric(df['YOM'], errors='coerce').fillna(0).astype(int).astype(str)
        # Revert 0 to something that can be handled or leave as '0' to be caught later
        df['YOM'] = df['YOM'].replace('0', '')
    
    if '< Back' in df.columns:
         df['< Back'] = df['< Back'].astype(str).replace('nan', '')

    # 2. Drop columns
    cols_to_drop = ['URL', 'Url', '< Back', 'Start Type', 'Get Leasing', 'Contact', 'Details']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_cols_to_drop:
        df.drop(columns=existing_cols_to_drop, inplace=True)
        logging.info(f"Dropped columns: {existing_cols_to_drop}")

    # 3. Clean 'Make'
    if 'Make' in df.columns:
        df["Make"] = df["Make"].astype(str).str.strip()
        df["Make"] = df["Make"].replace(["", "nan", "None"], np.nan)
        df.dropna(subset=["Make"], inplace=True)

    # 4. Clean 'Engine (cc)'
    engine_col = 'Engine (cc)'
    if engine_col in df.columns:
        df[engine_col] = df[engine_col].astype(str).str.strip()
        df[engine_col] = df[engine_col].replace(['-', ''], np.nan)
        df[engine_col] = pd.to_numeric(df[engine_col], errors='coerce')
        
        # Filter 1: Reasonable global limits
        df = df[
            (df[engine_col].isna()) | 
            ((df[engine_col] >= 500) & (df[engine_col] <= 8000))
        ]
        
        # Filter 2: Specific exclusion (Suzuki > 4000)
        df = df[~((df['Make'] == 'Suzuki') & (df[engine_col] > 4000))]
        
        # Filter 3: Final Strict Range
        df = df[
            (df[engine_col] >= 600) &
            (df[engine_col] < 3500)
        ]

    # 5. Clean 'Mileage (km)'
    mileage_col = 'Mileage (km)'
    if 'Mileage' in df.columns and mileage_col not in df.columns:
        df.rename(columns={'Mileage': mileage_col}, inplace=True)
        
    if mileage_col in df.columns:
        df[mileage_col] = df[mileage_col].astype(str).str.strip()
        df[mileage_col] = df[mileage_col].replace(["-", "nan"], np.nan)
        # Remove 'km' and commas if scraper didn't
        df[mileage_col] = df[mileage_col].str.replace("km", "", regex=False).str.replace(",", "", regex=False)
        df[mileage_col] = pd.to_numeric(df[mileage_col], errors="coerce")
        
        df = df[
            (df[mileage_col] > 0) & 
            (df[mileage_col] <= 500000)
        ]

    # 6. Clean 'Fuel Type'
    if 'Fuel Type' in df.columns:
        df["Fuel Type"] = df["Fuel Type"].astype(str).str.strip()
        df["Fuel Type"] = df["Fuel Type"].replace(["", "nan", "None"], np.nan)
        df["Fuel Type"] = df["Fuel Type"].replace({
            "Gas": "Other",
            "Electric": "Other"
        })

    # 7. Clean 'Gear'
    if 'Gear' in df.columns:
        df["Gear"] = df["Gear"].astype(str).str.strip()
        df["Gear"] = df["Gear"].replace(["", "nan", "None"], np.nan)
        df.dropna(subset=["Gear"], inplace=True)
        df = df[df["Gear"].isin(["Automatic", "Manual", "CVT"])]

    # 8. Clean 'YOM'
    if 'YOM' in df.columns:
        df['YOM'] = pd.to_numeric(df['YOM'], errors='coerce')
        # Apply Final Filter directly (>= 1990)
        df = df[df["YOM"] >= 1990]

    # 9. Feature Engineering
    logging.info("Feature Engineering...")
    
    # Vehicle Age
    if 'YOM' in df.columns:
        current_year = datetime.now().year
        df["Vehicle Age"] = current_year - df["YOM"]

    # Condition (extracted from Title)
    if 'Title' in df.columns:
        df["Condition"] = df["Title"].str.contains(
            "Unregistered", case=False, na=False
        ).map({True: "Unregistered", False: "Used"})
        df.drop(columns=['Title'], inplace=True)

    # Options Parsing
    if 'Options' in df.columns:
        df["Options"] = df["Options"].astype(str).str.strip()
        df["Options"] = df["Options"].replace("-", "")

        df["Has_AC"] = df["Options"].str.contains("AIR CONDITION", case=False).astype(int)
        df["Has_PowerSteering"] = df["Options"].str.contains("POWER STEERING", case=False).astype(int)
        df["Has_PowerMirror"] = df["Options"].str.contains("POWER MIRROR", case=False).astype(int)
        df["Has_PowerWindow"] = df["Options"].str.contains("POWER WINDOW", case=False).astype(int)

        df.drop(columns=["Options"], inplace=True)

        df["Option_Count"] = (
            df["Has_AC"] +
            df["Has_PowerSteering"] +
            df["Has_PowerMirror"] +
            df["Has_PowerWindow"]
        )

    # 10. Clean 'Price' & Duplicates
    if 'Price' in df.columns:
        # Remove 'Negotiable' and clean format
        df = df[~df["Price"].astype(str).str.contains("Negotiable", case=False, na=False)]
        
        df["Price"] = (
            df["Price"]
            .astype(str)
            .str.replace("Rs.", "", regex=False)
            .str.replace("/=", "", regex=False) # Helper for scraper variations
            .str.replace(",", "", regex=False)
            .str.replace(r"\(Ongoing Lease\)", "", regex=True)
            .str.strip()
        )
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df.dropna(subset=["Price"], inplace=True)

    # Deduplication
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    logging.info(f"Removed {initial_len - len(df)} duplicates.")
    
    logging.info(f"Shape after cleaning: {df.shape}")
    return df

def main():
    df = load_raw_data()
    if df is not None:
        cleaned_df = clean_data(df)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_listings_{timestamp}.csv"
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        cleaned_df.to_csv(filepath, index=False)
        logging.info(f"Saved cleaned data to {filepath}")
        
        # Also update the 'latest' processed file for easy access
        latest_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_listings_latest.csv")
        cleaned_df.to_csv(latest_path, index=False)
    else:
        logging.info("No data to clean.")

if __name__ == "__main__":
    main()
