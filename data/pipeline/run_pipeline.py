
import subprocess
import os
import sys
import logging
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
current_dir   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(os.path.dirname(current_dir))

# Define paths relative to Project Root
SCRAPER_SCRIPT  = os.path.join(PROJECT_ROOT, "scraping",  "scrapers",  "scrape_listings.py")
CLEANER_SCRIPT  = os.path.join(PROJECT_ROOT, "data",      "pipeline",  "clean_data.py")
ANOMALY_SCRIPT  = os.path.join(PROJECT_ROOT, "data",      "pipeline",  "detect_anomalies.py")
DRIFT_SCRIPT    = os.path.join(PROJECT_ROOT, "data",      "pipeline",  "detect_drift.py")
TRAINER_SCRIPT  = os.path.join(PROJECT_ROOT, "data",      "pipeline",  "train_model.py")
DRIFT_FLAG_FILE = os.path.join(PROJECT_ROOT, "data",      "processed", "drift_detected.flag")
BASELINE_CSV    = os.path.join(PROJECT_ROOT, "data",      "initial-cleaning", "cleaned-before_log.csv")
LATEST_CSV      = os.path.join(PROJECT_ROOT, "data",      "processed", "cleaned_listings_latest.csv")

# Read FORCE_RETRAIN from environment (set by GitHub Actions workflow_dispatch input)
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "false").strip().lower() == "true"

def run_script(script_path, name):
    logging.info(f"Starting {name}")
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        rc = process.poll()
        if rc == 0:
            logging.info(f"{name} completed successfully.")
            return True
        else:
            logging.error(f"{name} failed with error code {rc}")
            return False

    except Exception as e:
        logging.error(f"Error running {name}: {e}")
        return False

def main():
    start_time = datetime.now()
    logging.info(f"Pipeline started at {start_time}")
    logging.info(f"Project Root: {PROJECT_ROOT}")
    logging.info(f"FORCE_RETRAIN: {FORCE_RETRAIN}")

    # 1. Scraping
    if not run_script(SCRAPER_SCRIPT, "Scraper"):
        logging.error("Scraping failed. Aborting pipeline.")
        sys.exit(1)

    # 2. Anomaly Detection
    if not run_script(ANOMALY_SCRIPT, "Anomaly Detector"):
        logging.error("Anomaly Detection failed. Aborting pipeline.")
        sys.exit(1)

    # 3. Cleaning
    if not run_script(CLEANER_SCRIPT, "Data Cleaner"):
        logging.error("Cleaning failed. Aborting pipeline.")
        sys.exit(1)

    # 4. Drift Detection
    if not run_script(DRIFT_SCRIPT, "Drift Detector"):
        logging.error("Drift detection failed. Aborting pipeline.")
        sys.exit(1)


    # 5. Check whether retraining should happen
    drift_flagged = os.path.exists(DRIFT_FLAG_FILE)

    if drift_flagged:
        logging.info("Drift detected — proceeding to model retraining.")
    elif FORCE_RETRAIN:
        logging.info("FORCE_RETRAIN=true — triggering retraining regardless of drift.")
    else:
        logging.info("No drift detected and FORCE_RETRAIN=false. Skipping model retraining.")

    if drift_flagged or FORCE_RETRAIN:
        # ── Merge new scraped data with the historical baseline ──────────────
        # Retraining ALWAYS uses baseline + new data so the model retains
        # all historical market knowledge while also learning from new listings.
        # This avoids low-data overfitting when scraping only a few pages.
        try:
            import pandas as pd
            os.makedirs(os.path.dirname(LATEST_CSV), exist_ok=True)

            baseline_df = pd.read_csv(BASELINE_CSV)
            baseline_df.columns = [c.strip() for c in baseline_df.columns]
            logging.info(f"Baseline rows loaded: {len(baseline_df):,}")

            if os.path.exists(LATEST_CSV):
                new_df = pd.read_csv(LATEST_CSV)
                new_df.columns = [c.strip() for c in new_df.columns]
                logging.info(f"New scraped rows loaded: {len(new_df):,}")

                # Stack and deduplicate (new data takes priority on duplicates)
                combined = pd.concat([baseline_df, new_df], ignore_index=True)
                combined.drop_duplicates(inplace=True)
                logging.info(f"Combined dataset: {len(combined):,} rows "
                             f"(baseline {len(baseline_df):,} + new {len(new_df):,}, "
                             f"deduped {len(baseline_df)+len(new_df)-len(combined):,})")
            else:
                combined = baseline_df
                logging.info("No new scraped data found — retraining on baseline only.")

            combined.to_csv(LATEST_CSV, index=False)
            logging.info(f"Combined training data written to {LATEST_CSV}")

        except Exception as e:
            logging.error(f"Failed to merge datasets: {e}. Aborting retraining.")
            sys.exit(1)

        # ── Retrain the model ────────────────────────────────────────────────
        if not run_script(TRAINER_SCRIPT, "Model Trainer"):
            logging.error("Model training failed.")
            sys.exit(1)

        # Clean up drift flag if it was set
        if drift_flagged:
            try:
                os.remove(DRIFT_FLAG_FILE)
                logging.info("Drift flag removed.")
            except Exception as e:
                logging.warning(f"Could not remove drift flag: {e}")

    end_time = datetime.now()
    logging.info(f"Pipeline finished at {end_time} (Duration: {end_time - start_time})")

if __name__ == "__main__":
    main()
