"""
shape_analysis.py

Extract bounded design parameters from the component segmentation
produced by Stallion's trained YOLO segmentation model.

Expected component classes:

    back_cushion
    base
    left_arm
    legs
    right_arm
    seat_cushion

The module works in normalized image-space ratios.

It does NOT claim to recover true physical dimensions from a single
photograph. User-supplied L/W/H remain the authoritative dimensions
for the 3D CAD model.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import cv2
import numpy as np


DEFAULTS = {
    "arm_width_ratio": 0.095,
    "backrest_depth_ratio": 0.200,
    "seat_height_ratio": 0.470,
    "leg_height_ratio": 0.094,
    "arm_roundness": 0.55,
    "back_roundness": 0.55,
}


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

ALIASES = {
    "arm": {
        "arm",
        "armrest",
        "leftarm",
        "rightarm",
        "left_arm",
        "right_arm",
        "handle",
        "handles",
    },

    "back": {
        "back",
        "backrest",
        "back_rest",
        "backcushion",
        "back_cushion",
    },

    "seat": {
        "seat",
        "seatcushion",
        "seat_cushion",
    },

    "leg": {
        "leg",
        "legs",
        "foot",
        "feet",
    },
}


def _norm_label(label: Any) -> str:
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(label).lower(),
    )


def _match_group(label: str) -> Optional[str]:

    n = _norm_label(label)

    for group, aliases in ALIASES.items():

        normalized_aliases = {
            _norm_label(a)
            for a in aliases
        }

        if n in normalized_aliases:
            return group

    if "arm" in n or "handle" in n:
        return "arm"

    if "back" in n:
        return "back"

    if "seat" in n:
        return "seat"

    if "leg" in n or "foot" in n:
        return "leg"

    return None


# ---------------------------------------------------------------------------
# Array / mask helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()

    return np.asarray(x)


def _mask_from_detection(
    detection: Dict[str, Any],
    image_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Convert a validator detection's serialized mask into an image-sized
    uint8 mask.

    The validator stores masks as nested lists, so this works directly
    with the output of sofa_validator.py.
    """

    raw_mask = detection.get("mask")

    if raw_mask is None:
        return None

    try:
        mask = _to_numpy(raw_mask)

        # Remove unnecessary dimensions.
        mask = np.squeeze(mask)

        if mask.ndim != 2:
            return None

        mask = (mask > 0).astype(np.uint8)

    except Exception:
        return None

    h, w = image_shape

    if mask.shape != (h, w):

        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask


def _bbox(
    mask: np.ndarray,
) -> Optional[tuple[int, int, int, int]]:

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


# ---------------------------------------------------------------------------
# Roundness
# ---------------------------------------------------------------------------

def _roundness(mask: np.ndarray) -> Optional[float]:
    """
    Estimate silhouette roundness from bounding-box fill ratio.

    This is deliberately a proxy.

        0 ~= boxy / square
        1 ~= strongly rounded

    It should be used as a shape-control signal, not as a true
    physical fillet-radius measurement.
    """

    b = _bbox(mask)

    if b is None:
        return None

    x0, y0, x1, y1 = b

    bw = x1 - x0
    bh = y1 - y0

    if bw < 8 or bh < 8:
        return None

    crop = mask[y0:y1, x0:x1]

    area = float(
        (crop > 0).sum()
    )

    fill = area / float(
        bw * bh
    )

    score = (
        (0.96 - fill)
        / (0.96 - 0.68)
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


# ---------------------------------------------------------------------------
# Group validator detections
# ---------------------------------------------------------------------------

def _group_component_detections(
    component_detections: Any,
    image_shape: tuple[int, int],
) -> Dict[str, list[Dict[str, Any]]]:

    grouped = {
        "arm": [],
        "back": [],
        "seat": [],
        "leg": [],
    }

    if not component_detections:
        return grouped

    for detection in component_detections:

        if not isinstance(detection, dict):
            continue

        label = detection.get(
            "class_name",
            "",
        )

        group = _match_group(label)

        if group is None:
            continue

        mask = _mask_from_detection(
            detection,
            image_shape,
        )

        if mask is None:
            continue

        item = dict(detection)

        item["_mask"] = mask

        grouped[group].append(item)

    return grouped


# ---------------------------------------------------------------------------
# Duplicate / noisy arm handling
# ---------------------------------------------------------------------------

def _select_arm_masks(
    arms: list[Dict[str, Any]],
    image_width: int,
) -> list[np.ndarray]:
    """
    Handle duplicate arm detections.

    Prefer one left-most and one right-most arm region.

    If the model produces duplicate masks on the same side, keep the
    larger mask for that side.
    """

    if not arms:
        return []

    candidates = []

    for item in arms:

        mask = item["_mask"]

        b = _bbox(mask)

        if b is None:
            continue

        x0, y0, x1, y1 = b

        area = float(
            (mask > 0).sum()
        )

        center_x = (
            x0 + x1
        ) / 2.0

        candidates.append(
            {
                "mask": mask,
                "bbox": b,
                "area": area,
                "center_x": center_x,
            }
        )

    if not candidates:
        return []

    # Sort by horizontal position.
    candidates.sort(
        key=lambda x: x["center_x"]
    )

    # If only one candidate exists, keep it.
    if len(candidates) == 1:
        return [candidates[0]["mask"]]

    # Split candidates around image center.
    center = image_width / 2.0

    left = [
        c for c in candidates
        if c["center_x"] < center
    ]

    right = [
        c for c in candidates
        if c["center_x"] >= center
    ]

    selected = []

    if left:
        selected.append(
            max(
                left,
                key=lambda x: x["area"],
            )["mask"]
        )

    if right:
        selected.append(
            max(
                right,
                key=lambda x: x["area"],
            )["mask"]
        )

    return selected


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_component_detections(
    component_detections: Any,
    image_shape: tuple[int, int],
) -> Dict[str, Any]:
    """
    Main entry point for Stallion's validator output.

    Parameters
    ----------
    component_detections:
        The list from sofa_validator.py.

    image_shape:
        (height, width) of the original image.
    """

    h, w = image_shape

    grouped = _group_component_detections(
        component_detections,
        image_shape,
    )

    # ---------------------------------------------------------------
    # Build overall sofa silhouette
    # ---------------------------------------------------------------

    all_masks = [
        item["_mask"]
        for items in grouped.values()
        for item in items
    ]

    if not all_masks:

        return {
            "design_params": {},
            "confidence": 0.0,
            "measurements": {},
            "source": "yolo_segmentation",
        }

    union = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    for mask in all_masks:

        union = np.maximum(
            union,
            (mask > 0).astype(np.uint8),
        )

    sofa_box = _bbox(union)

    if sofa_box is None:

        return {
            "design_params": {},
            "confidence": 0.0,
            "measurements": {},
            "source": "yolo_segmentation",
        }

    sx0, sy0, sx1, sy1 = sofa_box

    sofa_w = max(
        sx1 - sx0,
        1,
    )

    sofa_h = max(
        sy1 - sy0,
        1,
    )

    params: Dict[str, float] = {}

    measurements: Dict[str, Any] = {
        "sofa_bbox_px": [
            sx0,
            sy0,
            sx1,
            sy1,
        ],

        "component_counts": {
            key: len(value)
            for key, value in grouped.items()
        },
    }

    # ---------------------------------------------------------------
    # ARMS
    # ---------------------------------------------------------------

    selected_arms = _select_arm_masks(
        grouped["arm"],
        image_width=w,
    )

    arm_widths = []

    for mask in selected_arms:

        b = _bbox(mask)

        if b:

            arm_widths.append(
                (b[2] - b[0])
                / sofa_w
            )

    if arm_widths:

        params["arm_width_ratio"] = float(
            np.clip(
                np.mean(arm_widths),
                0.045,
                0.18,
            )
        )

        measurements[
            "arm_width_ratios"
        ] = arm_widths

    # ---------------------------------------------------------------
    # SEAT HEIGHT
    # ---------------------------------------------------------------

    seat_boxes = []

    for item in grouped["seat"]:

        b = _bbox(
            item["_mask"]
        )

        if b:
            seat_boxes.append(b)

    if seat_boxes:

        seat_top = min(
            b[1]
            for b in seat_boxes
        )

        seat_height_ratio = (
            sy1 - seat_top
        ) / sofa_h

        params[
            "seat_height_ratio"
        ] = float(
            np.clip(
                seat_height_ratio,
                0.30,
                0.68,
            )
        )

        measurements[
            "seat_height_ratio_px"
        ] = params[
            "seat_height_ratio"
        ]

    # ---------------------------------------------------------------
    # LEGS
    # ---------------------------------------------------------------

    leg_heights = []

    for item in grouped["leg"]:

        b = _bbox(
            item["_mask"]
        )

        if b:

            leg_heights.append(
                (b[3] - b[1])
                / sofa_h
            )

    if leg_heights:

        params[
            "leg_height_ratio"
        ] = float(
            np.clip(
                np.median(
                    leg_heights
                ),
                0.035,
                0.20,
            )
        )

        measurements[
            "leg_height_ratios"
        ] = leg_heights

    # ---------------------------------------------------------------
    # BACKREST
    # ---------------------------------------------------------------

    back_boxes = []

    for item in grouped["back"]:

        b = _bbox(
            item["_mask"]
        )

        if b:
            back_boxes.append(b)

    if back_boxes:

        back_widths = [
            (b[2] - b[0])
            / sofa_w
            for b in back_boxes
        ]

        candidate = float(
            np.median(
                back_widths
            )
        )

        # A frontal back cushion can span most of the sofa width.
        # In that case its width is NOT a reliable depth measurement.
        if candidate < 0.38:

            params[
                "backrest_depth_ratio"
            ] = float(
                np.clip(
                    candidate,
                    0.10,
                    0.32,
                )
            )

            measurements[
                "backrest_depth_proxy"
            ] = candidate

    # ---------------------------------------------------------------
    # ARM ROUNDNESS
    # ---------------------------------------------------------------

    arm_roundness = []

    for mask in selected_arms:

        score = _roundness(mask)

        if score is not None:
            arm_roundness.append(score)

    if arm_roundness:

        params[
            "arm_roundness"
        ] = float(
            np.clip(
                np.mean(
                    arm_roundness
                ),
                0.0,
                1.0,
            )
        )

        measurements[
            "arm_roundness"
        ] = arm_roundness

    # ---------------------------------------------------------------
    # BACK ROUNDNESS
    # ---------------------------------------------------------------

    back_roundness = []

    for item in grouped["back"]:

        score = _roundness(
            item["_mask"]
        )

        if score is not None:
            back_roundness.append(score)

    if back_roundness:

        params[
            "back_roundness"
        ] = float(
            np.clip(
                np.mean(
                    back_roundness
                ),
                0.0,
                1.0,
            )
        )

        measurements[
            "back_roundness"
        ] = back_roundness

    # ---------------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------------

    confidence = float(
        min(
            1.0,
            len(params) / 6.0,
        )
    )

    return {
        "design_params": params,
        "confidence": confidence,
        "measurements": measurements,
        "source": "yolo_segmentation",
    }


# ---------------------------------------------------------------------------
# Compatibility function for raw YOLO results
# ---------------------------------------------------------------------------

def analyze_yolo_results(
    results: Any,
    image_shape: tuple[int, int],
) -> Dict[str, Any]:

    """
    Compatibility wrapper.

    Converts raw Ultralytics Results into the same structure expected
    by analyze_component_detections().
    """

    if results is None:

        return {
            "design_params": {},
            "confidence": 0.0,
            "measurements": {},
            "source": "yolo_segmentation",
        }

    if not isinstance(
        results,
        (list, tuple),
    ):
        results = [results]

    detections = []

    for result in results:

        names = getattr(
            result,
            "names",
            {},
        )

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        masks = getattr(
            result,
            "masks",
            None,
        )

        if boxes is None:
            continue

        cls = getattr(
            boxes,
            "cls",
            None,
        )

        if cls is None:
            continue

        cls_arr = _to_numpy(
            cls
        ).reshape(-1)

        for i, cls_id in enumerate(cls_arr):

            label = (
                names.get(
                    int(cls_id),
                    str(cls_id),
                )
                if isinstance(
                    names,
                    dict,
                )
                else str(cls_id)
            )

            group = _match_group(
                label
            )

            if group is None:
                continue

            if masks is None:
                continue

            data = getattr(
                masks,
                "data",
                None,
            )

            if data is None:
                continue

            arr = _to_numpy(
                data
            )

            if arr.ndim == 2:
                arr = arr[None, ...]

            if i >= len(arr):
                continue

            mask = (
                arr[i] > 0.5
            ).astype(
                "uint8"
            )

            h, w = image_shape

            if mask.shape != (h, w):

                mask = cv2.resize(
                    mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

            detections.append(
                {
                    "class_name": str(label),
                    "mask": mask.tolist(),
                }
            )

    return analyze_component_detections(
        detections,
        image_shape,
    )


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def analyze_image(
    image_path: str,
    detector: Any = None,
    yolo_results: Any = None,
    component_detections: Any = None,
) -> Dict[str, Any]:

    """
    Analyze an image.

    Preferred usage in Stallion pipeline:

        analyze_image(
            image_path,
            component_detections=analysis["component_detections"]
        )

    This avoids running YOLO twice.
    """

    image = cv2.imread(
        image_path
    )

    if image is None:

        return {
            "design_params": {},
            "confidence": 0.0,
            "measurements": {},
            "source": "yolo_segmentation",
        }

    h, w = image.shape[:2]

    if component_detections is not None:

        return analyze_component_detections(
            component_detections,
            (h, w),
        )

    if yolo_results is not None:

        return analyze_yolo_results(
            yolo_results,
            (h, w),
        )

    # Last-resort compatibility mode.
    if detector is not None:

        try:

            if hasattr(
                detector,
                "predict",
            ):

                yolo_results = detector.predict(
                    source=image_path,
                    verbose=False,
                )

            else:

                yolo_results = detector(
                    image_path
                )

            return analyze_yolo_results(
                yolo_results,
                (h, w),
            )

        except Exception:

            pass

    return {
        "design_params": {},
        "confidence": 0.0,
        "measurements": {},
        "source": "yolo_segmentation",
    }