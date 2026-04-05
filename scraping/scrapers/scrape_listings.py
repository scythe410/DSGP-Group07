"""
Scraper for riyasewana.com car listings.

Phase 1 — collect listing URLs from search results pages (li.v-card selector).
Phase 2 — visit each listing and extract structured data from .detail-row cards.

Always writes listings_latest.csv so downstream pipeline steps can rely on it
existing.  A timestamped archive copy is kept under data/raw/ for auditing.
"""

import os
import re
import time
import shutil
import logging
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_URL   = "https://riyasewana.com/search/cars"
START_PAGE = 1
END_PAGE   = 4      # 4 pages ≈ 160 listings; raise for full historical runs
MAX_ADS    = 80     # cap per pipeline run to fit within CI time budget

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_ARCHIVE = os.path.join(DATA_DIR, f'listings_{TIMESTAMP}.csv')
LATEST_CSV  = os.path.join(DATA_DIR, 'listings_latest.csv')

# Columns expected by the cleaning / training pipeline
COLUMNS = ['Url', 'Title', 'Price', 'Make', 'Model', 'YOM',
           'Mileage (km)', 'Engine (cc)', 'Gear', 'Fuel Type',
           'Condition', 'Location']


# ── Driver setup ───────────────────────────────────────────────────────────────
def setup_driver():
    opts = Options()
    opts.add_argument('--headless')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--disable-gpu')
    opts.add_argument('--disable-blink-features=AutomationControlled')
    opts.add_experimental_option('excludeSwitches', ['enable-automation'])
    opts.add_experimental_option('useAutomationExtension', False)
    # Disable image loading for speed
    opts.add_experimental_option('prefs',
        {'profile.managed_default_content_settings.images': 2})
    opts.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    svc = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=svc, options=opts)
    # Hide automation flag
    driver.execute_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    return driver


# ── Phase 1: collect listing URLs ──────────────────────────────────────────────
def collect_links(driver):
    """Return a deduplicated list of listing URLs scraped from search pages."""
    links = []
    seen  = set()
    for page_num in range(START_PAGE, END_PAGE + 1):
        url = f"{BASE_URL}?page={page_num}"
        log.info(f"Scanning page {page_num}: {url}")
        driver.get(url)
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        page_links = 0
        for card in soup.select('li.v-card'):
            a = card.select_one('.v-card-img a[href]')
            if not a:
                continue
            href = a['href'].strip()
            if not href.startswith('http'):
                href = 'https:' + href
            if href not in seen:
                seen.add(href)
                links.append(href)
                page_links += 1

        log.info(f"  → {page_links} new links on page {page_num} "
                 f"({len(links)} total so far)")
    return links


# ── Phase 2: scrape an individual listing page ─────────────────────────────────
def scrape_listing(url, driver):
    """
    Scrape a single listing page using the new riyasewana.com layout.

    Key selectors (as of 2026-04):
      Price   : div.price-amount
      Details : div.detail-row > span.detail-label + span.detail-value
      Title   : h1
    """
    try:
        driver.get(url)
        time.sleep(1)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        data = {'Url': url}

        # Title
        h1 = soup.select_one('h1')
        data['Title'] = h1.get_text(strip=True) if h1 else ''

        # Price
        price_el = soup.select_one('.price-amount')
        data['Price'] = price_el.get_text(strip=True) if price_el else ''

        # Structured detail rows
        for row in soup.select('.detail-row'):
            label_el = row.select_one('.detail-label')
            value_el = row.select_one('.detail-value')
            if not label_el or not value_el:
                continue
            label = label_el.get_text(strip=True)
            value = value_el.get_text(strip=True)

            # Map site label names → our column names
            if label == 'Mileage':
                data['Mileage (km)'] = value
            elif label == 'Year':
                data['YOM'] = value   # pipeline expects YOM (Osanda's feature_engineering.py)
            elif label in ('Make', 'Model', 'Gear', 'Fuel Type',
                           'Engine (cc)', 'Condition', 'Location'):
                data[label] = value

        return data

    except Exception as e:
        log.warning(f"Error scraping {url}: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    driver = setup_driver()
    try:
        # ── Phase 1 ──────────────────────────────────────────────────────────
        log.info(f"PHASE 1: Collecting links (pages {START_PAGE}–{END_PAGE})")
        links = collect_links(driver)
        log.info(f"Total links collected: {len(links)}")

        if not links:
            debug_path = os.path.join(DATA_DIR, 'debug_page1.html')
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            log.warning(f"0 ads found — page source saved to {debug_path}")
            log.warning("Probable cause: Cloudflare bot detection or HTML structure change.")
            # Write empty CSV so the pipeline exits cleanly (no false drift)
            pd.DataFrame(columns=COLUMNS).to_csv(LATEST_CSV, index=False)
            return

        if len(links) > MAX_ADS:
            log.info(f"Capping at {MAX_ADS} ads (pipeline time budget)")
            links = links[:MAX_ADS]

        # ── Phase 2 ──────────────────────────────────────────────────────────
        log.info(f"PHASE 2: Scraping {len(links)} listings")
        records = []
        for i, url in enumerate(links, 1):
            log.info(f"  [{i}/{len(links)}] {url}")
            rec = scrape_listing(url, driver)
            if rec:
                records.append(rec)
                log.info(f"    → {rec.get('Make','')} {rec.get('Model','')} "
                         f"{rec.get('YOM','')} | {rec.get('Price','')} | "
                         f"Mileage: {rec.get('Mileage (km)','N/A')}")

        if not records:
            log.warning("All individual page scrapes failed — writing empty CSV.")
            pd.DataFrame(columns=COLUMNS).to_csv(LATEST_CSV, index=False)
            return

        df = pd.DataFrame(records, columns=COLUMNS)
        log.info(f"\nDONE — {len(df)} records scraped successfully")
        df.to_csv(CSV_ARCHIVE, index=False)
        shutil.copy2(CSV_ARCHIVE, LATEST_CSV)
        log.info(f"Archive : {CSV_ARCHIVE}")
        log.info(f"Canonical: {LATEST_CSV}")

    except Exception as e:
        log.error(f"CRITICAL ERROR: {e}")
        raise
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
