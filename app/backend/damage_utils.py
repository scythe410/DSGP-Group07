"""
damage_utils.py — Pure helper functions for the dual-signal damage detection algorithm.

Extracted from api.py so they can be imported and unit-tested independently
of FastAPI, PyTorch, YOLO, or any other framework dependency.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Repair cost tiers based on SegFormer pixel area AS A PERCENTAGE of total image.
# Using percentage makes tiers resolution-independent (phone photos, webcam, etc.)
# Each entry: (max_damage_pct, repair_action, cost_lkr). None = catch-all.
# ─────────────────────────────────────────────────────────────────────────────
REPAIR_TIERS = [
    (1.5,  "Paintless Dent Repair",        5_000),   # <1.5%  — small ding/hail dent
    (6.0,  "Panel Beating",               12_000),   # <6.0%  — moderate dent, 1 panel
    (None, "Panel Replacement + Repaint", 35_000),   # ≥6.0%  — severe / multi-panel
]

# Minimum SegFormer pixel area to count as real surface damage (filters out noise).
MIN_SEGFORMER_DAMAGE_PX = 500


def estimate_repair(dent_area: int, total_pixels: int) -> tuple:
    """
    Maps SegFormer pixel area to a repair action and cost.
    Uses percentage of image area so results are resolution-independent.

    Returns:
        (repair_action: str, cost_lkr: int, damage_pct: float)
    """
    damage_pct = (dent_area / total_pixels * 100) if total_pixels > 0 else 0.0
    for max_pct, repair, cost in REPAIR_TIERS:
        if max_pct is None or damage_pct < max_pct:
            return repair, cost, round(damage_pct, 2)
    return REPAIR_TIERS[-1][1], REPAIR_TIERS[-1][2], round(damage_pct, 2)


def filter_detections_by_mask(detailed_detections: list, mask) -> tuple:
    """
    Filters YOLO detections to only those whose bounding box has ≥20% overlap
    with the SegFormer damage mask. Eliminates false positives scattered across
    the image when the real damage is in a specific region.

    Args:
        detailed_detections: list of {"class", "confidence", "box": [x1,y1,x2,y2]}
        mask: boolean H×W numpy array from SegFormer, or None if unavailable

    Returns:
        (filtered_groups: set[str], filtered_detections: list)
    """
    if mask is None or not detailed_detections:
        return {d['class'] for d in detailed_detections}, detailed_detections

    h, w = mask.shape
    filtered = []
    for det in detailed_detections:
        x1, y1, x2, y2 = [int(v) for v in det['box']]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area == 0:
            continue
        overlap_pixels = int(np.sum(mask[y1:y2, x1:x2]))
        if overlap_pixels / box_area >= 0.20:
            filtered.append(det)

    return {d['class'] for d in filtered}, filtered
