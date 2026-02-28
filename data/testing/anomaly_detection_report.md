# Anomaly Detection Subsystem: Testing & Evaluation Report

## 1. Problem Statement
Automated web scraping is inherently brittle. Structural changes on target sites, human error during data entry (e.g., misplacing zeros in prices), and temporary network glitches can produce erroneous or "garbage" data. If this anomalous data is ingested into the automated data pipeline, it will poison the downstream XGBoost predictive price model, degrading its accuracy over time.

To maintain pipeline integrity, a pre-cleaning Anomaly Detection module was necessary to flag and quarantine anomalous listings mathematically before formal data cleaning.

## 2. Methodology & Model Candidates
Since we do not have a curated set of "labeled anomalies" (ground truth), we employed an Unsupervised Machine Learning strategy. We formulated a "Bake-Off" testing approach evaluating three models on the same historical baseline dataset (11,441 records):

*   **Isolation Forest (IF)**: Works by isolating data points through random splits. It is historically excellent at catching extreme, sparse outliers.
*   **One-Class SVM (OCSVM)**: Learns a tight decision boundary around dense regions of "normal" data. It is theoretically better at catching nuanced anomalies that fall near the edge of standard conditional distributions.
*   **Local Outlier Factor (LOF)**: Measures local density deviation. Best used for offline structural analysis rather than point-in-time deployment for new single transactions, as standard Sklearn implementations of LOF cannot `.predict()` on novel unseen rows without refitting the entire historical set.

## 3. Experimental Setup & Synthetic Injections
The features extracted and scaled for the algorithm were `Price`, `Mileage (km)`, and `Engine (cc)`. 

To effectively test recall on models without ground-truth anomaly labels, we injected **Synthetic Tricky Test Cases** representing edge artifacts common to the complicated Sri Lankan vehicle market:

1.  **Obvious Garbage**: A generic Suzuki Alto priced at 50,000,000 LKR with an 800cc engine.
2.  **Mileage Discrepancy (Condition Mismatch)**: A WagonR priced normally at 3,500,000 LKR, but with an impossible physically driven mileage of 1,500,000 km.
3.  **Severe Underpricing**: A brand new Sedan (0 km, 1500cc) listed for only 150,000 LKR (1.5 Lakhs).

## 4. Results & Analysis

### 4.1 Global Flag Rates
Running on the 11,441 baseline records at a contamination rate of `0.01` (1%), the models identified similar global counts:
*   **Isolation Forest**: Flagged 115 rows
*   **One-Class SVM**: Flagged 121 rows
*   **LOF**: Flagged 115 rows

*(The actual flagged anomalies are retained in CSV formats within the `data/testing` folder for reference.)*

### 4.2 Synthetic Injection Results
The determining factor for model selection was how the models categorized the synthetic injections.

| Test Case | Description | Isolation Forest Result | One-Class SVM Result |
| :--- | :--- | :--- | :--- |
| **Test 1** | Obvious Garbage (50M LKR Alto) | **CAUGHT** | **CAUGHT** |
| **Test 2** | Insane Mileage, Normal Price | *MISSED* | **CAUGHT** |
| **Test 3** | Severe Underpricing (1.5 Lakhs) | *MISSED* | *MISSED* |

### 4.3 Observations
*   **Extreme Outliers**: Both IF and SVM successfully detected massive, one-dimensional scaling errors (Test 1).
*   **Feature Combination Anomalies**: Isolation Forest missed the high-mileage anomaly (Test 2) because a 1.5M km mileage and a 3.5M LKR price are not "extreme outliers" individually across the entire dataset (e.g., heavy machinery might have high mileage, luxury cars have high prices). However, **One-Class SVM** recognized that the *combination* of that price and that mileage fell outside the hypersphere of normal data density, effectively catching the mismatch.
*   **Categorical Context Limits**: Both models failed Test 3 (the 1.5 Lakh car). This exposes a limitation of using purely numerical Unsupervised ML. A brand new 1.5 Lakh string of variables mathematically matches a motorized tricycle (tuk-tuk) or a standard motorcycle perfectly. Because the algorithms were denied categorical context (`Make` and `Model`), they correctly identified the variables as "normal," unaware that it was mislabeled as a sedan.

## 5. Conclusions & Deployment Decision
**One-Class SVM** proved conclusively superior in identifying complex conditional mismatches on continuous variables in the Sri Lankan market space. While it suffers from the same categorical-blindness as Isolation Forest, its tighter dimensional boundary fitting makes it far safer as an automated pipeline gatekeeper.

**Final Action**: 
*   `detect_anomalies.py` was constructed using the optimized `One-Class SVM` algorithm configuration (`nu=0.01`, `kernel='rbf'`). 
*   It has been placed immediately after the `scrape_listings.py` extraction stage in the `run_pipeline.py` orchestrator. 
*   Any rows flagged as `-1` (anomalous) are written out to a quarantine log (`quarantined_ads.csv`) for optional human review, preventing downstream data contamination.
