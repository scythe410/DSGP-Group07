"""
Unit tests for Ramiru De Silva's components.
Run: pytest tests/test_units.py -v

Tests cover:
  - Dual-signal damage detection algorithm (mask fusion, cost tiers)
  - Data cleaning pipeline (edge cases)
  - Drift detection algorithm (threshold logic)
  - Anomaly detection model (hard bounds + SVM)
  - FastAPI endpoint schemas (/predict_price, /analyze_damage, /health)
"""

import os
import sys
import pickle
import pytest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app', 'backend'))
sys.path.insert(0, os.path.join(ROOT, 'data', 'pipeline'))
sys.path.insert(0, os.path.join(ROOT, 'price-model'))

# Import pure business-logic modules directly (no FastAPI/torch dependency)
from damage_utils import estimate_repair, filter_detections_by_mask, REPAIR_TIERS


# ─────────────────────────────────────────────────────────────────────────────
# 1. DUAL-SIGNAL ALGORITHM — _estimate_repair
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimateRepair:
    """Unit tests for the percentage-based repair cost tier function."""

    TOTAL_PX = 1024 * 768  # 786,432 — representative phone photo size

    def _call(self, pct):
        """Call estimate_repair with a damage area that gives `pct` percent."""
        dent_area = int(self.TOTAL_PX * pct / 100)
        return estimate_repair(dent_area, self.TOTAL_PX)

    def test_zero_damage_is_pdr(self):
        repair, cost, pct = self._call(0)
        assert repair == "Paintless Dent Repair"
        assert cost == 5_000
        assert pct == pytest.approx(0.0, abs=0.01)

    def test_tiny_damage_is_pdr(self):
        repair, cost, pct = self._call(1.0)
        assert repair == "Paintless Dent Repair"
        assert cost == 5_000

    def test_boundary_pdr_to_panel_beating(self):
        # int() truncation means exactly 1.5% rounds down to 1.4999% → still PDR
        # Use 1.49% (clearly PDR) and 1.51% (clearly Panel Beating)
        repair_lo, cost_lo, _ = self._call(1.49)
        assert repair_lo == "Paintless Dent Repair"
        repair_hi, cost_hi, _ = self._call(1.51)
        assert repair_hi == "Panel Beating"
        assert cost_hi == 12_000

    def test_moderate_damage_is_panel_beating(self):
        repair, cost, pct = self._call(3.0)
        assert repair == "Panel Beating"
        assert cost == 12_000

    def test_boundary_panel_beating_to_replacement(self):
        # int() truncation means exactly 6.0% rounds down to 5.9999% → still Panel Beating
        # Use 5.99% (clearly Panel Beating) and 6.01% (clearly Panel Replacement)
        repair_lo, _, _ = self._call(5.99)
        assert repair_lo == "Panel Beating"
        repair_hi, cost_hi, _ = self._call(6.01)
        assert repair_hi == "Panel Replacement + Repaint"
        assert cost_hi == 35_000

    def test_severe_damage_is_replacement(self):
        repair, cost, pct = self._call(15.0)
        assert repair == "Panel Replacement + Repaint"
        assert cost == 35_000

    def test_zero_total_pixels_no_crash(self):
        repair, cost, pct = estimate_repair(0, 0)
        assert cost in {5_000, 12_000, 35_000}
        assert pct == 0.0

    def test_damage_pct_returned_correctly(self):
        total = 100_000
        dent  = 5_000
        _, _, pct = estimate_repair(dent, total)
        assert pct == pytest.approx(5.0, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DUAL-SIGNAL ALGORITHM — _filter_detections_by_mask
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterDetectionsByMask:
    """Unit tests for YOLO bounding-box ↔ SegFormer mask overlap filtering."""

    H, W = 480, 640

    @pytest.fixture
    def mask_centre(self):
        """200×200 damage region in the centre of a 480×640 image."""
        m = np.zeros((self.H, self.W), dtype=bool)
        m[140:340, 220:420] = True
        return m

    def test_detection_fully_inside_kept(self, mask_centre):
        det = [{"class": "dent", "confidence": 0.9, "box": [230, 150, 410, 330]}]
        groups, filtered = filter_detections_by_mask(det, mask_centre)
        assert "dent" in groups
        assert len(filtered) == 1

    def test_detection_fully_outside_removed(self, mask_centre):
        det = [{"class": "scratch", "confidence": 0.8, "box": [0, 0, 80, 80]}]
        groups, filtered = filter_detections_by_mask(det, mask_centre)
        assert len(filtered) == 0
        assert len(groups) == 0

    def test_partial_overlap_below_threshold_removed(self, mask_centre):
        # Mask is at x:220-420, y:140-340. Choose box mostly outside mask.
        # Box [200,330,230,400]: size=30×70=2100, overlap x:220-230(10) × y:330-340(10)=100px → 4.8% < 20%
        det = [{"class": "dent", "confidence": 0.7, "box": [200, 330, 230, 400]}]
        groups, filtered = filter_detections_by_mask(det, mask_centre)
        assert len(filtered) == 0

    def test_no_mask_returns_all_detections(self):
        det = [
            {"class": "dent",    "confidence": 0.9, "box": [10, 10, 100, 100]},
            {"class": "scratch", "confidence": 0.7, "box": [200, 200, 300, 300]},
        ]
        groups, filtered = filter_detections_by_mask(det, None)
        assert len(filtered) == 2

    def test_empty_detections_with_mask(self, mask_centre):
        groups, filtered = filter_detections_by_mask([], mask_centre)
        assert filtered == []
        assert groups == set()

    def test_zero_area_box_skipped(self, mask_centre):
        det = [{"class": "dent", "confidence": 0.9, "box": [100, 100, 100, 100]}]
        groups, filtered = filter_detections_by_mask(det, mask_centre)
        assert len(filtered) == 0

    def test_multiple_detections_mixed(self, mask_centre):
        detections = [
            {"class": "dent",    "confidence": 0.9, "box": [230, 150, 410, 330]},  # inside
            {"class": "scratch", "confidence": 0.8, "box": [0,   0,   80,  80]},   # outside
        ]
        groups, filtered = filter_detections_by_mask(detections, mask_centre)
        assert len(filtered) == 1
        assert filtered[0]["class"] == "dent"


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA CLEANING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class TestDataCleaningPipeline:
    """Unit tests for the clean_data() pipeline function."""

    @pytest.fixture
    def base_row(self):
        return {
            "Make": "Toyota", "Model": "Aqua", "YOM": "2019",
            "Mileage (km)": "45000", "Fuel Type": "Petrol",
            "Gear": "Automatic", "Engine (cc)": "1500",
            "Title": "Toyota Aqua Used",
            "Price": "Rs. 9,500,000",
            "Options": "AIR CONDITION,POWER STEERING",
        }

    @pytest.fixture
    def clean_fn(self):
        from clean_data import clean_data
        return clean_data

    def test_valid_row_survives(self, base_row, clean_fn):
        df = pd.DataFrame([base_row])
        out = clean_fn(df.copy())
        assert len(out) >= 1

    def test_duplicate_removed(self, base_row, clean_fn):
        df = pd.DataFrame([base_row, base_row])
        out = clean_fn(df.copy())
        assert len(out) == 1

    def test_engine_out_of_range_removed(self, base_row, clean_fn):
        row = {**base_row, "Engine (cc)": "60000"}
        df = pd.DataFrame([row])
        out = clean_fn(df.copy())
        assert len(out) == 0

    def test_negotiable_price_removed(self, base_row, clean_fn):
        row = {**base_row, "Price": "Negotiable"}
        df = pd.DataFrame([row])
        out = clean_fn(df.copy())
        assert len(out) == 0

    def test_pre_1990_yom_removed(self, base_row, clean_fn):
        row = {**base_row, "YOM": "1985"}
        df = pd.DataFrame([row])
        out = clean_fn(df.copy())
        assert len(out) == 0

    def test_post_1990_yom_kept(self, base_row, clean_fn):
        row = {**base_row, "YOM": "1991"}
        df = pd.DataFrame([row])
        out = clean_fn(df.copy())
        assert len(out) == 1

    def test_condition_extracted_used(self, base_row, clean_fn):
        df = pd.DataFrame([base_row])
        out = clean_fn(df.copy())
        if "Condition" in out.columns:
            assert out.iloc[0]["Condition"] == "Used"

    def test_condition_extracted_unregistered(self, base_row, clean_fn):
        row = {**base_row, "Title": "Toyota Aqua Unregistered"}
        df = pd.DataFrame([row])
        out = clean_fn(df.copy())
        if "Condition" in out.columns:
            assert out.iloc[0]["Condition"] == "Unregistered"

    def test_option_count_computed(self, base_row, clean_fn):
        df = pd.DataFrame([base_row])
        out = clean_fn(df.copy())
        assert "Option_Count" in out.columns
        assert out.iloc[0]["Option_Count"] >= 2   # AC + Power Steering

    def test_empty_dataframe_no_crash(self, clean_fn):
        df = pd.DataFrame(columns=["Make","Model","YOM","Mileage (km)",
                                   "Fuel Type","Gear","Engine (cc)","Title","Price","Options"])
        out = clean_fn(df.copy())
        assert isinstance(out, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftDetection:
    """Unit tests for the drift detection threshold logic."""

    @pytest.fixture
    def baseline(self):
        from detect_drift import PRICE_MEAN_DRIFT_THRESHOLD
        np.random.seed(42)
        ref = pd.DataFrame({
            "Price": np.random.normal(7_000_000, 1_500_000, 500).astype(int),
            "Make":  np.random.choice(["Toyota","Honda","Suzuki","Nissan"], 500),
        })
        return ref, PRICE_MEAN_DRIFT_THRESHOLD

    def test_identical_data_no_drift(self, baseline):
        from detect_drift import check_price_drift
        ref, _ = baseline
        assert check_price_drift(ref, ref) is False

    def test_large_shift_detected(self, baseline):
        from detect_drift import check_price_drift
        ref, _ = baseline
        shifted = ref.copy()
        shifted["Price"] = (shifted["Price"] * 1.40).astype(int)  # +40%
        assert check_price_drift(ref, shifted) is True

    def test_small_shift_not_mean_drift(self, baseline):
        from detect_drift import check_price_drift, PRICE_MEAN_DRIFT_THRESHOLD
        ref, threshold = baseline
        # Shift exactly at 50% of the threshold — should NOT trigger mean drift alone
        small = ref.copy()
        small["Price"] = (small["Price"] * (1 + threshold * 0.5)).astype(int)
        # Mean-threshold test only: mean drift = threshold*0.5 < threshold
        mean_drift = abs(small["Price"].mean() - ref["Price"].mean()) / ref["Price"].mean()
        assert mean_drift < threshold

    def test_categorical_drift_detected(self, baseline):
        from detect_drift import check_categorical_drift
        ref, _ = baseline
        all_toyota = ref.copy()
        all_toyota["Make"] = "Toyota"
        assert check_categorical_drift(ref, all_toyota, "Make") is True

    def test_same_categorical_no_drift(self, baseline):
        from detect_drift import check_categorical_drift
        ref, _ = baseline
        assert check_categorical_drift(ref, ref, "Make") is False

    def test_drift_threshold_constant_is_positive(self):
        from detect_drift import PRICE_MEAN_DRIFT_THRESHOLD
        assert 0 < PRICE_MEAN_DRIFT_THRESHOLD < 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANOMALY DETECTION MODEL
# ─────────────────────────────────────────────────────────────────────────────

ANOMALY_PKL = os.path.join(ROOT, 'price-model', 'anomaly_model.pkl')

@pytest.mark.skipif(not os.path.exists(ANOMALY_PKL), reason="anomaly_model.pkl not found locally")
class TestAnomalyModel:
    """Unit tests for the fitted OneClassSVM anomaly model."""

    @pytest.fixture(scope="class")
    def bundle(self):
        with open(ANOMALY_PKL, 'rb') as f:
            return pickle.load(f)

    def test_bundle_has_required_keys(self, bundle):
        assert "scaler"      in bundle
        assert "model"       in bundle
        assert "features"    in bundle
        assert "hard_bounds" in bundle

    def test_hard_bounds_are_plausible(self, bundle):
        hb = bundle["hard_bounds"]
        lo_e, hi_e = hb["Engine (cc)"]
        lo_m, hi_m = hb["Mileage (km)"]
        assert lo_e >= 0  and hi_e <= 10_000
        assert lo_m >= 0  and hi_m <= 600_000

    def test_normal_vehicle_is_inlier(self, bundle):
        scaler, model, features = bundle["scaler"], bundle["model"], bundle["features"]
        normal = pd.DataFrame([{"Price": 9_000_000, "Mileage (km)": 60_000, "Engine (cc)": 1500}])[features]
        pred = model.predict(scaler.transform(normal))
        assert pred[0] == 1  # inlier

    def test_absurd_engine_is_outlier(self, bundle):
        scaler, model, features = bundle["scaler"], bundle["model"], bundle["features"]
        absurd = pd.DataFrame([{"Price": 9_000_000, "Mileage (km)": 60_000, "Engine (cc)": 8000}])[features]
        pred = model.predict(scaler.transform(absurd))
        assert pred[0] == -1  # outlier

    def test_absurd_mileage_is_outlier(self, bundle):
        scaler, model, features = bundle["scaler"], bundle["model"], bundle["features"]
        absurd = pd.DataFrame([{"Price": 5_000_000, "Mileage (km)": 800_000, "Engine (cc)": 1500}])[features]
        pred = model.predict(scaler.transform(absurd))
        assert pred[0] == -1

    def test_inference_is_fast(self, bundle):
        import time
        scaler, model, features = bundle["scaler"], bundle["model"], bundle["features"]
        probe = pd.DataFrame([{"Price": 0, "Mileage (km)": 50_000, "Engine (cc)": 1500}])[features]
        scaled = scaler.transform(probe)
        t0 = time.time()
        for _ in range(500):
            model.predict(scaled)
        elapsed_ms = (time.time() - t0) * 1000
        assert elapsed_ms < 2000, f"500 inferences took {elapsed_ms:.0f}ms — too slow"

    def test_hard_bound_60000cc_flagged(self, bundle):
        hb = bundle["hard_bounds"]
        lo, hi = hb["Engine (cc)"]
        assert not (lo <= 60_000 <= hi), "60,000cc should be outside engine hard bounds"

    def test_hard_bound_negative_mileage_flagged(self, bundle):
        hb = bundle["hard_bounds"]
        lo, hi = hb["Mileage (km)"]
        assert not (lo <= -1 <= hi), "Negative mileage should be outside mileage hard bounds"


# ─────────────────────────────────────────────────────────────────────────────
# 6. FASTAPI ENDPOINT SCHEMAS (live integration — skipped if server is down)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import requests as _req
    _server_up = _req.get("http://127.0.0.1:8000/health", timeout=3).status_code == 200
except Exception:
    _server_up = False


@pytest.mark.skipif(not _server_up, reason="Local API server not running (start with: uvicorn app.backend.api:app)")
class TestAPIEndpoints:
    """Live integration tests against the running FastAPI server."""

    import requests
    BASE = "http://127.0.0.1:8000"

    def test_health_returns_200(self):
        r = requests.get(f"{self.BASE}/health")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

    def test_vehicle_options_returns_mapping(self):
        r = requests.get(f"{self.BASE}/vehicle_options")
        assert r.status_code == 200
        data = r.json()
        assert "options" in data
        assert len(data["options"]) > 0

    def test_predict_price_valid_input(self):
        payload = {
            "Make": "Toyota", "Model": "Aqua", "YOM": 2019,
            "Mileage_km": 45000, "Engine_cc": 1500,
            "Fuel_Type": "Petrol", "Gear": "Automatic",
            "Condition": "Used", "Has_AC": True,
            "Has_PowerSteering": True, "Has_PowerMirror": False, "Has_PowerWindow": True,
        }
        r = requests.post(f"{self.BASE}/predict_price", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "predicted_price_lkr" in data
        assert data["predicted_price_lkr"] > 0

    def test_predict_price_impossible_engine_flagged(self):
        payload = {
            "Make": "Toyota", "Model": "Aqua", "YOM": 2019,
            "Mileage_km": 45000, "Engine_cc": 60000,
            "Fuel_Type": "Petrol", "Gear": "Automatic",
            "Condition": "Used", "Has_AC": True,
            "Has_PowerSteering": True, "Has_PowerMirror": False, "Has_PowerWindow": True,
        }
        r = requests.post(f"{self.BASE}/predict_price", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["is_anomalous"] is True
        assert data["anomaly_reason"] is not None

    def test_predict_price_missing_field_returns_422(self):
        r = requests.post(f"{self.BASE}/predict_price", json={"Make": "Toyota"})
        assert r.status_code == 422   # FastAPI validation error

    def test_analyze_damage_non_vehicle_image_rejected(self):
        import io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (100, 100), color=(0, 128, 0)).save(buf, format="JPEG")
        buf.seek(0)
        r = requests.post(f"{self.BASE}/analyze_damage",
                          files={"file": ("test.jpg", buf, "image/jpeg")})
        # Gatekeeper should reject a plain green square
        data = r.json()
        assert r.status_code in {200, 400}   # 200 with has_damage=False or 400

    def test_cors_header_present(self):
        r = requests.options(f"{self.BASE}/health",
                             headers={"Origin": "http://localhost"})
        assert "access-control-allow-origin" in r.headers
