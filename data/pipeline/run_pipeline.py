
import subprocess
import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(current_dir))

# Define paths relative to Project Root
SCRAPER_SCRIPT = os.path.join(PROJECT_ROOT, "scraping", "scrapers", "scrape_listings.py")
CLEANER_SCRIPT = os.path.join(PROJECT_ROOT, "data", "pipeline", "clean_data.py")
DRIFT_SCRIPT = os.path.join(PROJECT_ROOT, "data", "pipeline", "detect_drift.py")
TRAINER_SCRIPT = os.path.join(PROJECT_ROOT, "data", "pipeline", "train_model.py")
DRIFT_FLAG_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "drift_detected.flag")

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
            stderr_output = process.stderr.read() if process.stderr else ""
            logging.error(f"Stderr:\n{stderr_output}")
            return False

    except Exception as e:
        logging.error(f"Error running {name}: {e}")
        return False

def main():
    start_time = datetime.now()
    logging.info(f"Pipeline started at {start_time}")
    logging.info(f"Project Root detected as: {PROJECT_ROOT}")
    
    # 1. Scraping
    if not run_script(SCRAPER_SCRIPT, "Scraper"):
        logging.error("Scraping failed. Aborting pipeline.")
        sys.exit(1)
        
    # 2. Cleaning
    if not run_script(CLEANER_SCRIPT, "Data Cleaner"):
        logging.error("Cleaning failed. Aborting pipeline.")
        sys.exit(1)
        
    # 3. Drift Detection
    if not run_script(DRIFT_SCRIPT, "Drift Detector"):
        logging.error("Drift detection failed. Aborting pipeline.")
        sys.exit(1)
        
    # 4. Check for Drift Flag
    if os.path.exists(DRIFT_FLAG_FILE):
        logging.info("Drift detected! Proceeding to model retraining.")
        
        # 5. Retraining
        if not run_script(TRAINER_SCRIPT, "Model Trainer"):
            logging.error("Model training failed.")
            sys.exit(1)
            
        # Cleanup flag
        try:
            os.remove(DRIFT_FLAG_FILE)
            logging.info("Drift flag removed.")
        except Exception as e:
            logging.warning(f"Could not remove drift flag: {e}")
            
    else:
        logging.info("No drift detected. Skipping model retraining.")
        
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"Pipeline finished at {end_time} (Duration: {duration})")

if __name__ == "__main__":
    main()
