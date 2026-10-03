"""
sofa_style.py — Read the *look* of the customer's sofa from the photo so the
3D model resembles it, while the internal construction stays the industry
structure from sofa_structure.py.

Uses the component segmentation already produced by sofa_validator.py
(back_cushion, base, left_arm, right_arm, legs, seat_cushion). Everything
here is a bounded, best-effort estimate from one photo; missing evidence
falls back to the industry reference style.

    style = style_from_photo(image_path, analysis["component_detections"])
    Sofa3DGenerator(geo, request_id, sofa_type, style=style)

Returned keys (all optional for the 3D builder):
    arms           "both" | "none"
    arm_style      "sloped" (industry) | "track" (square) | "rolled"
    arm_top        arm top height / overall height
    seat_cushions  loose seat cushions (0 = tight seat)
    back_cushions  loose back cushions (0 = tight back)
    seat_top       seat top height / overall height
    leg_style      "tapered" (visible wooden legs) | "block" (hidden / plinth)
    fabric_rgb, leg_rgb
    evidence       what was measured, for display / debugging
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from shape_analysis import _mask_from_detection, _roundness

MIN_CONF = 0.45


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _dedupe(dets):
    """Keep confident detections, dropping heavy overlaps (duplicate boxes)."""
    keep = []
    for d in sorted(dets, key=lambda d: -d.get("confidence", 0)):
        if d.get("confidence", 0) < MIN_CONF:
            continue
        if all(_iou(d["bbox"], k["bbox"]) < 0.5 for k in keep):
            keep.append(d)
    return keep


def _median_rgb(image_bgr, mask, erode=3):
    if mask is None or mask.sum() < 50:
        return None
    m = cv2.erode(mask.astype(np.uint8), np.ones((erode, erode), np.uint8)) if erode else mask
    px = image_bgr[m > 0]
    if len(px) < 30:
        px = image_bgr[mask > 0]
    if len(px) == 0:
        return None
    # ignore blown highlights and deep shadow so the colour is the fabric's own
    lum = px.mean(1)
    lo, hi = np.percentile(lum, 15), np.percentile(lum, 90)
    sel = px[(lum >= lo) & (lum <= hi)]
    b, g, r = np.median(sel if len(sel) else px, axis=0)
    return int(r), int(g), int(b)


def _arm_style(mask, inner_is_right):
    """Classify one arm silhouette: rolled / sloped / track."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return None, {}
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    cols = np.arange(x0 + int(0.1 * w), x1 - int(0.1 * w) + 1)
    tops = []
    for c in cols:
        col = np.where(mask[:, c] > 0)[0]
        if len(col):
            tops.append(col.min())
    if len(tops) < 5:
        return None, {}
    tops = (np.array(tops, float) - y0) / h                # 0 = arm top, grows downward
    slope = np.polyfit(np.linspace(0, 1, len(tops)), tops, 1)[0]
    if not inner_is_right:
        slope = -slope                                     # + means inner side is lower
    rnd = _roundness(mask) or 0.0
    # curvature of the top edge: a roll bulges above the chord between its ends
    chord = np.linspace(tops[0], tops[-1], len(tops))
    bulge = float(np.clip((chord - tops).max(), 0, 1))
    return _classify_arm(bulge, slope), {"slope": round(float(slope), 3), "roundness": round(float(rnd), 3),
                                         "top_bulge": round(bulge, 3)}


def _classify_arm(bulge, slope):
    # Perspective makes single measurements noisy, so thresholds are
    # conservative and the caller decides on the median over both arms.
    if bulge > 0.12:
        return "rolled"
    if slope > 0.22:
        return "sloped"
    return "track"


def style_from_photo(image_path: str, component_detections: Any) -> dict:
    image = cv2.imread(str(image_path))
    if image is None or not component_detections:
        return {"evidence": {"note": "no segmentation — industry reference style used"}}
    H, W = image.shape[:2]

    by = {}
    for d in component_detections:
        by.setdefault(d.get("class_name"), []).append(d)
    for k in list(by):
        by[k] = _dedupe(by[k])

    def masks(cls):
        out = []
        for d in by.get(cls, []):
            m = _mask_from_detection(d, (H, W))
            if m is not None and m.sum() > 0:
                out.append((d, m))
        return out

    seat, back, base, legs = masks("seat_cushion"), masks("back_cushion"), masks("base"), masks("legs")
    arms = masks("left_arm") + masks("right_arm")

    union = np.zeros((H, W), np.uint8)
    for _, m in seat + back + base + legs + arms:
        union |= m
    if union.sum() == 0:
        return {"evidence": {"note": "empty masks — industry reference style used"}}
    ys, xs = np.where(union > 0)
    top_y, floor_y = ys.min(), ys.max()
    sofa_h = max(floor_y - top_y, 1)
    cx = (xs.min() + xs.max()) / 2

    style, ev = {}, {}

    # Fabric colour: seat + back + arm upholstery
    uph = np.zeros((H, W), np.uint8)
    for _, m in seat + back + arms:
        uph |= m
    rgb = _median_rgb(image, uph)
    if rgb:
        style["fabric_rgb"] = rgb

    # Arms
    if not arms and (seat or back):
        style["arms"] = "none"
    elif arms:
        style["arms"] = "both"
        votes, details, tops = [], [], []
        for _, m in arms:
            ay, ax = np.where(m > 0)
            inner_is_right = ax.mean() < cx                # arm on the image-left: inner side is its right
            s, info = _arm_style(m, inner_is_right)
            if s:
                votes.append(s)
                details.append(info)
            tops.append(ay.min())
        if votes:
            style["arm_style"] = _classify_arm(float(np.median([d["top_bulge"] for d in details])),
                                               float(np.median([d["slope"] for d in details])))
            ev["arm_shape"] = details
        style["arm_top"] = float(np.clip((floor_y - np.median(tops)) / sofa_h, 0.45, 0.97))

    # Cushions
    style["seat_cushions"] = len(seat)
    style["back_cushions"] = len(back)
    if seat:
        seat_top = min(np.where(m > 0)[0].min() for _, m in seat)
        style["seat_top"] = float(np.clip((floor_y - seat_top) / sofa_h, 0.3, 0.7))

    # Legs
    if legs:
        best = max(legs, key=lambda dm: dm[0].get("confidence", 0))
        style["leg_style"] = "tapered"
        # leg masks are loose: drop pixels that look like the backdrop or the upholstery
        border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]]).astype(float)
        bg = np.median(border, axis=0)
        px = image[best[1] > 0].astype(float)
        keep = np.linalg.norm(px - bg, axis=1) > 45
        mx, mn = px.max(1), px.min(1)
        sat = (mx - mn) / np.maximum(mx, 1)
        b_, g_, r_ = px[:, 0], px[:, 1], px[:, 2]
        # legs are neutral (black / metal / white) or warm wood (r >= g >= b)
        keep &= (sat < 0.2) | ((r_ >= g_ - 5) & (g_ >= b_ - 10))
        if rgb:
            fab_bgr = np.array(rgb[::-1], float)
            keep &= np.linalg.norm(px - fab_bgr, axis=1) > 35
        if keep.sum() >= 30:
            b, g, r = np.median(px[keep], axis=0)
            style["leg_rgb"] = (int(r), int(g), int(b))
    else:
        style["leg_style"] = "block"

    ev["counts"] = {k: len(v) for k, v in by.items()}
    style["evidence"] = ev
    return style


def describe(style: dict) -> str:
    """One-line human summary for the UI."""
    if not style or len(style) <= 1:
        return "Industry reference style (no photo evidence)"
    bits = []
    arms = style.get("arms", "both")
    bits.append("armless" if arms == "none" else f"{style.get('arm_style', 'sloped')} arms")
    sc, bc = style.get("seat_cushions", 0), style.get("back_cushions", 0)
    bits.append(f"{sc} seat cushion{'s' if sc != 1 else ''}" if sc else "tight seat")
    bits.append(f"{bc} back cushion{'s' if bc != 1 else ''}" if bc else "tight back")
    bits.append("wooden legs" if style.get("leg_style") == "tapered" else "hidden legs")
    return ", ".join(bits)
