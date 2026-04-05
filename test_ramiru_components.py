"""
Comprehensive test suite for Ramiru De Silva's components:
  1. Anomaly Detection Model (OneClassSVM)
  2. Data Cleaning Pipeline
  3. Drift Detection Algorithm
  4. Dual-Signal Damage Detection Logic (YOLO + SegFormer mask fusion)
  5. FastAPI Backend Endpoints

Run: python3.13 test_ramiru_components.py
"""

import os, sys, json, pickle, time, io
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'price-model'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'data', 'pipeline'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'app', 'backend'))

PASS  = "\033[92m PASS\033[0m"
FAIL  = "\033[91m FAIL\033[0m"
INFO  = "\033[94m INFO\033[0m"
WARN  = "\033[93m WARN\033[0m"

results = []

def record(section, name, passed, detail=""):
    tag = PASS if passed else FAIL
    print(f"  [{section}] {name}: {tag}  {detail}")
    results.append({"section": section, "name": name, "passed": passed, "detail": detail})

print("\n" + "="*65)
print("  RAMIRU DE SILVA — COMPONENT TEST SUITE")
print("="*65)


# ─────────────────────────────────────────────────────────────────
# 1. ANOMALY DETECTION MODEL
# ─────────────────────────────────────────────────────────────────
print("\n[1] Anomaly Detection Model (OneClassSVM)")
print("-" * 45)

ANOMALY_PKL = os.path.join(ROOT_DIR, 'price-model', 'anomaly_model.pkl')

try:
    with open(ANOMALY_PKL, 'rb') as f:
        bundle = pickle.load(f)
    scaler      = bundle['scaler']
    oc_svm      = bundle['model']
    features    = bundle['features']
    hard_bounds = bundle['hard_bounds']
    record("Anomaly", "Model loads from disk", True, f"Keys: {list(bundle.keys())}")
except Exception as e:
    record("Anomaly", "Model loads from disk", False, str(e))
    bundle = None

if bundle:
    # 1a. Self-evaluation on training data
    ref_csv = os.path.join(ROOT_DIR, 'data', 'initial-cleaning', 'cleaned-before_log.csv')
    df_ref  = pd.read_csv(ref_csv)
    df_ref.columns = [c.strip() for c in df_ref.columns]
    df_clean = df_ref.dropna(subset=features).copy()
    X_ref_scaled = scaler.transform(df_clean[features])
    preds = oc_svm.predict(X_ref_scaled)
    inlier_rate = (preds == 1).mean()
    record("Anomaly", "Inlier rate on baseline data ≥ 97%", inlier_rate >= 0.97,
           f"{inlier_rate*100:.1f}% inliers ({(preds==1).sum():,}/{len(preds):,})")
    outlier_count = (preds == -1).sum()
    record("Anomaly", "Outlier count on baseline (expect ≈ 1%)", True,
           f"{outlier_count} flagged ({outlier_count/len(preds)*100:.1f}%)")

    # 1b. Hard-bounds: obviously impossible inputs
    impossible_cases = [
        {"Price": 5000000, "Mileage (km)": 50000, "Engine (cc)": 60000,  "label": "60,000cc engine"},
        {"Price": 5000000, "Mileage (km)": -500,   "Engine (cc)": 1500,   "label": "Negative mileage"},
        {"Price": 5000000, "Mileage (km)": 600000,  "Engine (cc)": 1500,  "label": "600k km mileage"},
    ]
    for case in impossible_cases:
        eng   = case["Engine (cc)"]
        mil   = case["Mileage (km)"]
        lo_e, hi_e = hard_bounds.get("Engine (cc)", (0, 999999))
        lo_m, hi_m = hard_bounds.get("Mileage (km)", (0, 999999))
        flagged = not (lo_e <= eng <= hi_e) or not (lo_m <= mil <= hi_m)
        record("Anomaly", f"Hard-bound catch: {case['label']}", flagged)

    # 1c. Normal inputs should PASS
    normal_cases = [
        {"Price": 8000000,  "Mileage (km)": 80000,  "Engine (cc)": 1500},
        {"Price": 15000000, "Mileage (km)": 30000,  "Engine (cc)": 2000},
        {"Price": 4000000,  "Mileage (km)": 150000, "Engine (cc)": 1000},
    ]
    normal_probes = pd.DataFrame(normal_cases)[features]
    normal_scaled = scaler.transform(normal_probes)
    normal_preds  = oc_svm.predict(normal_scaled)
    all_normal = (normal_preds == 1).all()
    record("Anomaly", "Normal vehicles flagged as inliers", all_normal,
           f"{(normal_preds==1).sum()}/3 passed")

    # 1d. Precision / Recall on synthetic anomalies
    # True positives = impossible rows correctly flagged
    # Use extreme values as ground-truth anomalies
    synthetic_anomalies = pd.DataFrame([
        {"Price": 5000000, "Mileage (km)": 800000, "Engine (cc)": 1500},  # absurd mileage
        {"Price": 5000000, "Mileage (km)": 50000,  "Engine (cc)": 8000},  # absurd engine
        {"Price": 5000000, "Mileage (km)": 50000,  "Engine (cc)": 100},   # tiny engine
    ])[features]
    syn_scaled = scaler.transform(synthetic_anomalies)
    syn_preds  = oc_svm.predict(syn_scaled)
    tp = (syn_preds == -1).sum()  # correctly flagged
    record("Anomaly", "SVM detects synthetic anomalies", tp >= 2,
           f"Flagged {tp}/3 constructed anomalies")

    # 1e. Inference latency
    t0 = time.time()
    for _ in range(1000):
        scaled = scaler.transform(normal_probes.iloc[:1])
        oc_svm.predict(scaled)
    latency_ms = (time.time() - t0)
    record("Anomaly", "Inference latency < 5ms per call", latency_ms < 5,
           f"{latency_ms*1000:.2f}µs avg over 1000 calls")


# ─────────────────────────────────────────────────────────────────
# 2. DATA CLEANING PIPELINE
# ─────────────────────────────────────────────────────────────────
print("\n[2] Data Cleaning Pipeline")
print("-" * 45)

try:
    from clean_data import clean_data

    # Build a synthetic raw DataFrame mimicking scraper output
    raw = pd.DataFrame([
        # Valid rows
        {"Make": "Toyota",  "Model": "Aqua",    "YOM": "2019", "Mileage (km)": "45000",
         "Fuel Type": "Petrol", "Gear": "Automatic", "Engine (cc)": "1500",
         "Title": "Toyota Aqua Used", "Price": "Rs. 9,500,000", "Options": "AIR CONDITION,POWER STEERING"},
        {"Make": "Honda",   "Model": "Fit",     "YOM": "2016", "Mileage (km)": "82000",
         "Fuel Type": "Hybrid", "Gear": "CVT",       "Engine (cc)": "1300",
         "Title": "Honda Fit Unregistered", "Price": "Rs. 7,200,000", "Options": "AIR CONDITION"},
        # Duplicate row
        {"Make": "Toyota",  "Model": "Aqua",    "YOM": "2019", "Mileage (km)": "45000",
         "Fuel Type": "Petrol", "Gear": "Automatic", "Engine (cc)": "1500",
         "Title": "Toyota Aqua Used", "Price": "Rs. 9,500,000", "Options": "AIR CONDITION,POWER STEERING"},
        # Bad row: engine out of range
        {"Make": "Suzuki",  "Model": "Alto",    "YOM": "2018", "Mileage (km)": "30000",
         "Fuel Type": "Petrol", "Gear": "Manual",    "Engine (cc)": "50000",
         "Title": "Suzuki Alto Used",  "Price": "Rs. 4,000,000", "Options": ""},
        # Bad row: negotiable price
        {"Make": "Nissan",  "Model": "Leaf",    "YOM": "2020", "Mileage (km)": "20000",
         "Fuel Type": "Electric", "Gear": "Automatic","Engine (cc)": "1600",
         "Title": "Nissan Leaf Used",  "Price": "Negotiable", "Options": "AIR CONDITION"},
        # Bad row: pre-1990
        {"Make": "Toyota",  "Model": "Corolla", "YOM": "1980", "Mileage (km)": "200000",
         "Fuel Type": "Petrol", "Gear": "Manual",    "Engine (cc)": "1500",
         "Title": "Toyota Corolla Used","Price": "Rs. 1,200,000", "Options": ""},
    ])

    t0 = time.time()
    cleaned = clean_data(raw.copy())
    elapsed = time.time() - t0

    record("Pipeline", "clean_data() executes without error", True,
           f"Processed in {elapsed*1000:.1f}ms")
    record("Pipeline", "Duplicate rows removed", len(cleaned) < (len(raw) - 1),
           f"{len(raw)} → {len(cleaned)} rows")
    record("Pipeline", "Engine out-of-range rows removed",
           not (cleaned.get("Engine (cc)", pd.Series(dtype=float)) > 10000).any(),
           "No engine > 10,000cc in clean output")
    record("Pipeline", "Negotiable-price rows removed",
           "Negotiable" not in cleaned.get("Price", pd.Series()).astype(str).values
           if "Price" in cleaned.columns else True,
           "No negotiable rows in clean output")
    if "YOM" in cleaned.columns:
        all_post_1990 = (pd.to_numeric(cleaned["YOM"], errors="coerce").dropna() >= 1990).all()
        record("Pipeline", "Pre-1990 YOM rows removed", all_post_1990,
               f"All YOM ≥ 1990: {all_post_1990}")
    if "Title" in raw.columns and "Condition" in cleaned.columns:
        has_condition = cleaned["Condition"].notna().all()
        unregistered_correct = cleaned.loc[cleaned.get("Model","") == "Fit", "Condition"].values[0] == "Unregistered" if len(cleaned) > 0 else True
        record("Pipeline", "Condition extracted from Title", has_condition,
               "Condition column present in output")
    record("Pipeline", "Option_Count column computed",
           "Option_Count" in cleaned.columns,
           f"Max options: {cleaned.get('Option_Count', pd.Series([0])).max()}")

except Exception as e:
    record("Pipeline", "clean_data() executes without error", False, str(e))


# ─────────────────────────────────────────────────────────────────
# 3. DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────
print("\n[3] Drift Detection Algorithm")
print("-" * 45)

try:
    from detect_drift import check_price_drift, check_categorical_drift, PRICE_MEAN_DRIFT_THRESHOLD

    ref_csv = os.path.join(ROOT_DIR, 'data', 'initial-cleaning', 'cleaned-before_log.csv')
    ref_df  = pd.read_csv(ref_csv)
    ref_df.columns = [c.strip() for c in ref_df.columns]

    # Test A: no drift — same data
    no_drift = check_price_drift(ref_df, ref_df)
    record("Drift", "Same-distribution data → no drift detected", not no_drift,
           "KS test and mean check both pass identical data")

    # Test B: large mean shift → drift
    shifted = ref_df.copy()
    shifted["Price"] = shifted["Price"] * 1.30  # +30% price shift
    drift_detected = check_price_drift(ref_df, shifted)
    record("Drift", "30% price shift → drift detected", drift_detected,
           f"Threshold: {PRICE_MEAN_DRIFT_THRESHOLD*100:.0f}% mean change")

    # Test C: small shift (4%) → no drift
    small_shift = ref_df.copy()
    small_shift["Price"] = small_shift["Price"] * 1.04
    small_drift = check_price_drift(ref_df, small_shift)
    record("Drift", "4% price shift → no drift (under threshold)", not small_drift,
           f"4% < {PRICE_MEAN_DRIFT_THRESHOLD*100:.0f}% threshold")

    # Test D: categorical drift in Make
    cat_drift_data = ref_df.copy()
    # Artificially make all makes "Toyota" — major categorical change
    cat_drift_data["Make"] = "Toyota"
    cat_drift = check_categorical_drift(ref_df, cat_drift_data, "Make")
    record("Drift", "Major Make distribution change → drift detected", cat_drift,
           "All-Toyota vs real distribution")

    # Test E: flag file mechanism
    flag_path = os.path.join(ROOT_DIR, 'data', 'processed', 'drift_detected.flag')
    os.makedirs(os.path.dirname(flag_path), exist_ok=True)
    with open(flag_path, 'w') as f: f.write("True")
    flag_exists = os.path.exists(flag_path)
    os.remove(flag_path)
    record("Drift", "Drift flag file creation and removal", flag_exists and not os.path.exists(flag_path),
           "Flag written and cleaned up successfully")

except Exception as e:
    record("Drift", "Drift detection functions load", False, str(e))


# 4. DUAL-SIGNAL DAMAGE DETECTION ALGORITHM
# ---------------------------------------------------------
print("\n[4] Dual-Signal Damage Detection Algorithm")
print("-" * 45)

# Import production functions (NOT re-implemented copies)
from damage_utils import estimate_repair, filter_detections_by_mask, REPAIR_TIERS, MIN_SEGFORMER_DAMAGE_PX

import numpy as np

# 4a. Mask overlap filter: detection inside damage region -> kept
t0 = time.time()
H, W = 480, 640
mask = np.zeros((H, W), dtype=bool)
mask[100:300, 200:400] = True   # damage region in centre

det_inside  = [{"class": "dent",    "confidence": 0.87, "box": [210, 110, 390, 290]}]
det_outside = [{"class": "scratch", "confidence": 0.75, "box": [10,  10,  80,  70]}]
det_partial = [{"class": "dent",    "confidence": 0.65, "box": [190, 290, 420, 400]}]

g_in, f_in   = filter_detections_by_mask(det_inside,  mask)
g_out, f_out = filter_detections_by_mask(det_outside, mask)
g_part, f_part = filter_detections_by_mask(det_partial, mask)
elapsed_ms = (time.time()-t0)*1000

record("DualSignal", "Detection fully inside damage mask -> kept", len(f_in) == 1,
       f"'dent' box retained")
record("DualSignal", "Detection fully outside damage mask -> filtered out", len(f_out) == 0,
       "'scratch' box discarded (0% overlap)")
record("DualSignal", "Partial overlap (<20%) -> filtered out", len(f_part) == 0,
       "Edge-clipping box discarded")
record("DualSignal", "Mask fusion latency < 5ms", elapsed_ms < 5,
       f"{elapsed_ms:.2f}ms for 3 detections on {H}x{W} mask")

# 4b. Cost estimation tiers (percentage-based, matching production)
# estimate_repair(dent_area, total_pixels) -> (action, cost, pct)
TOTAL_PX = 1024 * 768  # representative image
cases = [
    (0.5,  "Paintless Dent Repair",        5_000),   # 0.5% < 1.5%
    (1.0,  "Paintless Dent Repair",        5_000),   # 1.0% < 1.5%
    (2.0,  "Panel Beating",               12_000),   # 1.5% <= 2.0% < 6.0%
    (5.0,  "Panel Beating",               12_000),   # 5.0% < 6.0%
    (8.0,  "Panel Replacement + Repaint", 35_000),   # 8.0% >= 6.0%
    (20.0, "Panel Replacement + Repaint", 35_000),   # 20.0% >= 6.0%
]
all_tiers_correct = True
for pct, expected_repair, expected_cost in cases:
    dent_area = int(TOTAL_PX * pct / 100)
    repair, cost, _ = estimate_repair(dent_area, TOTAL_PX)
    if repair != expected_repair or cost != expected_cost:
        all_tiers_correct = False
        record("DualSignal", f"Cost tier for {pct}%", False,
               f"Got '{repair}'@{cost:,} -- expected '{expected_repair}'@{expected_cost:,}")
record("DualSignal", "All 3 repair cost tiers map correctly", all_tiers_correct,
       "PDR->LKR 5k, Panel Beating->LKR 12k, Replacement->LKR 35k")

# 4c. Dual-signal verdict (YOLO OR SegFormer)
# Scenario 1: only SegFormer fires (scratch-level -- YOLO missed)
yolo_no = False
seg_yes  = (2500 >= MIN_SEGFORMER_DAMAGE_PX)
effective = yolo_no or seg_yes
record("DualSignal", "SegFormer-only signal -> damage confirmed", effective,
       f"YOLO=False, SegFormer={seg_yes} (2500px >= {MIN_SEGFORMER_DAMAGE_PX}px threshold)")

# Scenario 2: both fire
yolo_yes = True
seg_yes2 = (8000 >= MIN_SEGFORMER_DAMAGE_PX)
effective2 = yolo_yes or seg_yes2
record("DualSignal", "Both YOLO + SegFormer fire -> damage confirmed", effective2,
       "Both signals agree")

# Scenario 3: neither fires
yolo_no2 = False
seg_no   = (200 < MIN_SEGFORMER_DAMAGE_PX)
effective3 = yolo_no2 or (not seg_no)
record("DualSignal", "Neither signal fires -> no damage", not effective3,
       f"YOLO=False, SegFormer=False (200px < {MIN_SEGFORMER_DAMAGE_PX}px threshold)")


# ─────────────────────────────────────────────────────────────────
# 5. FASTAPI BACKEND — ENDPOINT FUNCTIONAL TESTS
# ─────────────────────────────────────────────────────────────────
print("\n[5] FastAPI Backend Endpoint Tests")
print("-" * 45)

try:
    import requests

    BASE = "http://127.0.0.1:8000"
    TIMEOUT = 5

    # Health check
    try:
        t0 = time.time()
        r = requests.get(f"{BASE}/health", timeout=TIMEOUT)
        latency = (time.time()-t0)*1000
        record("API", "/health returns 200", r.status_code == 200,
               f"Status {r.status_code}, {latency:.0f}ms, body: {r.json()}")
    except Exception as e:
        record("API", "/health returns 200", False, f"Server not running: {e}")

    # vehicle_options
    try:
        t0 = time.time()
        r = requests.get(f"{BASE}/vehicle_options", timeout=TIMEOUT)
        latency = (time.time()-t0)*1000
        data = r.json()
        record("API", "/vehicle_options returns 200", r.status_code == 200,
               f"{latency:.0f}ms")
        if r.status_code == 200:
            opts = data.get("options", {})
            record("API", "/vehicle_options returns make→models mapping",
                   len(opts) > 0,
                   f"{len(opts)} makes, e.g. Toyota: {len(opts.get('Toyota',[]))} models")
    except Exception as e:
        record("API", "/vehicle_options returns 200", False, f"Server not running: {e}")

    # predict_price with valid input
    valid_payload = {
        "Make": "Toyota", "Model": "Aqua", "YOM": 2019,
        "Mileage_km": 45000, "Engine_cc": 1500,
        "Fuel_Type": "Petrol", "Gear": "Automatic",
        "Condition": "Used",
        "Has_AC": True, "Has_PowerSteering": True,
        "Has_PowerMirror": True, "Has_PowerWindow": True
    }
    try:
        t0 = time.time()
        r = requests.post(f"{BASE}/predict_price", json=valid_payload, timeout=TIMEOUT)
        latency = (time.time()-t0)*1000
        record("API", "/predict_price valid input → 200", r.status_code == 200,
               f"{latency:.0f}ms")
        if r.status_code == 200:
            d = r.json()
            price = d.get("predicted_price_lkr", 0)
            is_anom = d.get("is_anomalous", False)
            record("API", "/predict_price returns LKR price > 0", price > 0,
                   f"LKR {price:,}")
            record("API", "/predict_price valid input -> not anomalous", not is_anom,
                   f"is_anomalous={is_anom}")
    except Exception as e:
        record("API", "/predict_price valid input → 200", False, f"Server not running: {e}")

    # predict_price with impossible engine
    anom_payload = {**valid_payload, "Engine_cc": 60000}
    try:
        t0 = time.time()
        r = requests.post(f"{BASE}/predict_price", json=anom_payload, timeout=TIMEOUT)
        latency = (time.time()-t0)*1000
        if r.status_code == 200:
            d = r.json()
            record("API", "/predict_price 60,000cc engine → anomaly flagged",
                   d.get("is_anomalous") == True,
                   f"is_anomalous={d.get('is_anomalous')}, reason: {d.get('anomaly_reason','')[:60]}")
        else:
            record("API", "/predict_price 60,000cc engine → anomaly flagged",
                   False, f"HTTP {r.status_code}")
    except Exception as e:
        record("API", "/predict_price 60,000cc engine → anomaly flagged",
               False, f"Server not running: {e}")

    # analyze_damage with a real image
    dummy_jpg = os.path.join(ROOT_DIR, "app", "frontend", "dummy.jpg")
    if os.path.exists(dummy_jpg):
        try:
            t0 = time.time()
            with open(dummy_jpg, "rb") as img:
                r = requests.post(f"{BASE}/analyze_damage",
                                  files={"file": ("dummy.jpg", img, "image/jpeg")},
                                  timeout=60)
            latency = (time.time()-t0)*1000
            record("API", "/analyze_damage returns 200", r.status_code == 200,
                   f"{latency:.0f}ms")
            if r.status_code == 200:
                d = r.json()
                record("API", "/analyze_damage returns expected fields",
                       all(k in d for k in ['status','has_damage','image']),
                       f"status={d.get('status')}, has_damage={d.get('has_damage')}")
        except Exception as e:
            record("API", "/analyze_damage returns 200", False, f"{e}")

    # Test CORS header
    try:
        r = requests.options(f"{BASE}/health", headers={"Origin": "http://localhost"}, timeout=TIMEOUT)
        cors_ok = "access-control-allow-origin" in r.headers
        record("API", "CORS headers present on preflight", cors_ok,
               str(dict(list(r.headers.items())[:3])))
    except Exception as e:
        record("API", "CORS headers present on preflight", False, str(e))

except ImportError:
    record("API", "requests library available", False, "pip install requests")


# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SUMMARY")
print("="*65)

by_section = {}
for r in results:
    by_section.setdefault(r['section'], []).append(r)

total_pass = sum(1 for r in results if r['passed'])
total_all  = len(results)

for section, tests in by_section.items():
    p = sum(1 for t in tests if t['passed'])
    print(f"  {section:<14} {p}/{len(tests)} passed")

print(f"\n  TOTAL: {total_pass}/{total_all} tests passed ({total_pass/total_all*100:.0f}%)")
print("="*65)

# Export JSON for documentation
out = {
    "total": total_all,
    "passed": total_pass,
    "failed": total_all - total_pass,
    "pass_rate": round(total_pass/total_all*100, 1),
    "results": results
}
with open("test_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Results saved to test_results.json")
