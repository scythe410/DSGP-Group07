
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
BASE_URL = "https://riyasewana.com/search/cars"
# Demo: 4 pages (~220 listings, ~15 min). Full run: END_PAGE = 664 (~110k listings, ~6 hrs)
START_PAGE = 1
END_PAGE = 4

# Directory Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Timestamped archive (for historical records)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = os.path.join(DATA_DIR, f'listings_{TIMESTAMP}.csv')
# Canonical "latest" output that the downstream pipeline always reads
LATEST_CSV  = os.path.join(DATA_DIR, 'listings_latest.csv')

def setup_driver():
    """Initializes Selenium with IMAGE LOADING DISABLED for speed."""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')

    # DISABLE IMAGES
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)

    # Up-to-date user agent matching current LTS Chrome
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scrape_vehicle_details(url, driver):
    """Scrapes a single page."""
    try:
        driver.get(url)
        # Sleep time
        time.sleep(1)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        vehicle_data = {'Url': url}

        # Title
        title = soup.find('h1')
        vehicle_data['Title'] = title.get_text(strip=True) if title else 'N/A'

        # Contact & Price
        spans = soup.find_all('span', class_='moreph')
        if len(spans) >= 2:
            vehicle_data['Contact'] = spans[0].get_text(strip=True)
            vehicle_data['Price'] = spans[1].get_text(strip=True)
        elif len(spans) == 1:
             # Heuristic: if it looks like a price (digits/currency), it's price
             text = spans[0].get_text(strip=True)
             if any(char.isdigit() for char in text):
                 vehicle_data['Price'] = text
             else:
                 vehicle_data['Contact'] = text

        # Details Table
        details_table = soup.find('table', class_='moret')
        if details_table:
            for row in details_table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) == 4:
                    vehicle_data[cells[0].get_text(strip=True).replace(":", "")] = cells[1].get_text(strip=True)
                    vehicle_data[cells[2].get_text(strip=True).replace(":", "")] = cells[3].get_text(strip=True)
                elif len(cells) == 2:
                    vehicle_data[cells[0].get_text(strip=True).replace(":", "")] = cells[1].get_text(strip=True)
        return vehicle_data
    except Exception as e:
        print(f"Error on {url}: {e}")
        return None

def main():
    driver = setup_driver()
    all_ad_links = []

    try:
        # Collect Links
        print(f"PHASE 1: Collecting links from page {START_PAGE} to {END_PAGE}")
        for page_num in range(START_PAGE, END_PAGE + 1):
            search_url = f"{BASE_URL}?page={page_num}"
            print(f"Scanning Page {page_num}...")

            driver.get(search_url)
            time.sleep(2)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            links_on_page = soup.select('li.item h2.more a')

            for link in links_on_page:
                href = link.get('href')
                if href: all_ad_links.append(href)

        print(f"Total Ads Found: {len(all_ad_links)}")

        if len(all_ad_links) == 0:
            # Save page source so we can debug bot-blocking / selector changes
            debug_path = os.path.join(DATA_DIR, 'debug_page1.html')
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            print(f"  [DEBUG] 0 ads found — page source saved to {debug_path}")
            print("  [WARN] Possible bot detection or riyasewana.com HTML change.")
            # Write empty listings_latest.csv so downstream steps skip gracefully
            pd.DataFrame(columns=['Url','Title','Contact','Price','Make','Model',
                                   'Mileage (km)','Engine (cc)','Gear','Fuel Type',
                                   'Condition']).to_csv(LATEST_CSV, index=False)
            print(f"  [INFO] Empty listings_latest.csv written — pipeline will skip retrain.")
            return

        # Limit for test purposes if too many
        if len(all_ad_links) > 20:
            print("Limiting to first 10 ads for demonstration speed...")
            all_ad_links = all_ad_links[:10]

        # Scrape & Save Incrementally
        print(f"PHASE 2: Scraping Details")

        scraped_buffer = []

        for i, ad_url in enumerate(all_ad_links):
            print(f"Scraping {i+1}/{len(all_ad_links)}: {ad_url}")
            data = scrape_vehicle_details(ad_url, driver)

            if data:
                print(f"  > Found: {data.get('Title', 'N/A')} - {data.get('Price', 'N/A')}")
                scraped_buffer.append(data)

            # CHECKPOINT: Save every 5 ads
            if len(scraped_buffer) >= 5:
                df_batch = pd.DataFrame(scraped_buffer)

                file_exists = os.path.isfile(CSV_FILENAME)
                df_batch.to_csv(CSV_FILENAME, mode='a', header=not file_exists, index=False)
                print(f"  [Saved batch to {CSV_FILENAME}]")
                scraped_buffer = []

        # Save any remaining data in the buffer
        if scraped_buffer:
            df_batch = pd.DataFrame(scraped_buffer)
            file_exists = os.path.isfile(CSV_FILENAME)
            df_batch.to_csv(CSV_FILENAME, mode='a', header=not file_exists, index=False)
            print(f"  [Saved final records to {CSV_FILENAME}]")

        print("\nDONE!")

        # Also write to the canonical 'listings_latest.csv' that the pipeline reads
        import shutil
        shutil.copy2(CSV_FILENAME, LATEST_CSV)
        print(f"  [INFO] Copied to {LATEST_CSV}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
