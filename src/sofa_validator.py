"""
Phase 3 - Sofa Image Validation (Plugin Architecture)

HOW IT WORKS
============
Detection is split into two layers:

  1. DETECTOR (pluggable backend)
     ─ Responsible for: "is there a sofa in this image, and where?"
     ─ Returns a list of raw detections: [{bbox, confidence}, ...]
     ─ Current default: YOLOv8DetectorBackend  (yolov8n.pt, COCO pre-trained)
     ─ Swap in your own model: implement SofaDetectorBase and pass it to validate_sofa()

  2. CLASSIFIER (shared logic, not pluggable yet)
     ─ Responsible for: "what type of sofa is this?"
     ─ Uses bbox aspect ratio + detection count as heuristics:
         2+ non-overlapping detections          →  l_shape
         1 detection, width/height < 1.30       →  1_seater
         1 detection, width/height 1.30–1.60    →  2_seater
         1 detection, width/height 1.60–3.20    →  3_seater   ✅ only supported
         1 detection, width/height ≥ 3.20       →  4_seater_plus

SWAPPING IN YOUR OWN MODEL (PLUGIN GUIDE)
==========================================
1. Create a new file, e.g.  src/my_sofa_model.py
2. Subclass SofaDetectorBase:

    from sofa_validator import SofaDetectorBase

    class MySofaDetector(SofaDetectorBase):
        def __init__(self, weights_path):
            import torch
            self.model = torch.load(weights_path)

        def detect(self, image_path: str) -> list[dict]:
            # Run your model on image_path.
            # Return a list of dicts, each with:
            #   "bbox":       [x1, y1, x2, y2]  (pixel coords, ints)
            #   "confidence": float  (0.0 – 1.0)
            # Return [] if no sofa is found.
            ...

3. Pass it into validate_sofa():

    from my_sofa_model import MySofaDetector
    from sofa_validator import validate_sofa

    detector = MySofaDetector("path/to/weights.pt")
    result = validate_sofa(image_path, request_folder, detector=detector)

4. Done — the classifier, gate logic, annotation, and JSON output all stay the same.
"""

import os
import json
from abc import ABC, abstractmethod

import cv2

# ---------------------------------------------------------------------------
# Constants (shared by all backends)
# ---------------------------------------------------------------------------

MIN_CONFIDENCE = 0.25   # detections below this are ignored

# Aspect-ratio thresholds (width ÷ height of bounding box)
# Calibrated against real sofa photography:
#   armchair   ~0.8–1.3   (nearly square)
#   2-seater   ~1.3–1.6   (moderately wide)
#   3-seater   ~1.6–3.2   (wide; upper bound is generous for low-res images)
#   4-seater+  >3.2       (panoramic or very large sofa)
_RATIO_RULES = [
    ("1_seater",      0.00,  1.30),
    ("2_seater",      1.30,  1.60),
    ("3_seater",      1.60,  3.20),
    ("4_seater_plus", 3.20,  float("inf")),
]

SUPPORTED_TYPE = "3_seater"   # only type allowed through to the cost engine

_TYPE_LABELS = {
    "1_seater":      "1-seater (armchair)",
    "2_seater":      "2-seater (loveseat)",
    "3_seater":      "3-seater",
    "4_seater_plus": "4-seater / large sofa",
    "l_shape":       "L-shape / sectional",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SofaValidationError(ValueError):
    """Raised when an image fails sofa validation (no sofa, or unsupported type)."""


# ---------------------------------------------------------------------------
# PLUGIN BASE CLASS  ← implement this to swap in your own model
# ---------------------------------------------------------------------------

class SofaDetectorBase(ABC):
    """
    Abstract interface every sofa detector backend must implement.

    A backend is responsible ONLY for finding sofas in an image.
    Classification (seater type), gating, and output formatting are
    handled by the shared validate_sofa() function above.
    """

    @abstractmethod
    def detect(self, image_path: str) -> list:
        """
        Run inference on *image_path*.

        Returns a list of detections, each a dict:
            {
                "bbox":       [x1, y1, x2, y2],   # pixel coords, ints
                "confidence": float,                # 0.0 – 1.0
            }

        Return an empty list [] if no sofa is found.
        Raise FileNotFoundError if the image doesn't exist.
        """


# ---------------------------------------------------------------------------
# BUILT-IN BACKEND: YOLOv8-nano  (default)
# ---------------------------------------------------------------------------

class YOLOv8DetectorBackend(SofaDetectorBase):
    """
    Default backend using YOLOv8-nano (pre-trained on COCO).

    COCO class 57 = "couch". Model weights (~6 MB) are downloaded
    automatically on first use from ultralytics' GitHub releases.

    Replace this class with your own SofaDetectorBase subclass when
    you have a custom-trained model.
    """

    COUCH_CLASS_ID = 57          # COCO index for "couch"
    MODEL_NAME     = "yolov8n.pt"

    def __init__(self):
        from ultralytics import YOLO
        self._model = YOLO(self.MODEL_NAME, verbose=False)

    def detect(self, image_path: str) -> list:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self._model(image_path, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id == self.COUCH_CLASS_ID and conf >= MIN_CONFIDENCE:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox":       [round(x1), round(y1), round(x2), round(y2)],
                    "confidence": round(conf, 4),
                })

        # Return sorted by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections


# ---------------------------------------------------------------------------
# PLUGIN PLACEHOLDER  ← drop your custom model in here
# ---------------------------------------------------------------------------

class CustomSofaDetectorPlaceholder(SofaDetectorBase):
    """
    ╔══════════════════════════════════════════════════════════╗
    ║  PLUGIN SLOT — Replace this with your own trained model  ║
    ╚══════════════════════════════════════════════════════════╝

    Steps to activate:
      1. Train your sofa classification model (YOLO, ResNet, etc.)
      2. Save the weights to  models/custom_sofa_model.pt  (or any path)
      3. Implement detect() below using your model's inference API
      4. In intake.py, change:
             from sofa_validator import validate_sofa
             result = validate_sofa(image_path, request_folder)
         to:
             from sofa_validator import validate_sofa, CustomSofaDetectorPlaceholder
             detector = CustomSofaDetectorPlaceholder("models/custom_sofa_model.pt")
             result = validate_sofa(image_path, request_folder, detector=detector)

    Your detect() output format (must match exactly):
        [
            {"bbox": [x1, y1, x2, y2], "confidence": 0.94},
            {"bbox": [x1, y1, x2, y2], "confidence": 0.61},  # optional 2nd detection
        ]

    Optionally, your model may also return a "sofa_type" key directly
    (e.g. "3_seater") — if present, the aspect-ratio classifier is skipped:
        [
            {"bbox": [...], "confidence": 0.94, "sofa_type": "3_seater"},
        ]
    """

    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path
        # TODO: load your model here, e.g.:
        # import torch
        # self.model = torch.load(weights_path)

    def detect(self, image_path: str) -> list:
        # TODO: implement inference with your model and return detections
        # in the format described above.
        raise NotImplementedError(
            "CustomSofaDetectorPlaceholder is not implemented yet. "
            "Follow the instructions in this class's docstring to plug in your model."
        )


# ---------------------------------------------------------------------------
# Shared helpers (used by all backends)
# ---------------------------------------------------------------------------

def _iou(box_a, box_b):
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _classify_type(detections):
    """
    Classify sofa type from detections.
    If a detection already carries "sofa_type" (from a custom model), use it directly.
    Otherwise fall back to bbox aspect-ratio heuristic.
    """
    if len(detections) >= 2:
        d0, d1 = detections[0], detections[1]
        if _iou(d0["bbox"], d1["bbox"]) < 0.40:
            return "l_shape"

    best = detections[0]

    # Custom model may return sofa_type directly — skip ratio logic
    if "sofa_type" in best:
        return best["sofa_type"]

    x1, y1, x2, y2 = best["bbox"]
    width  = x2 - x1
    height = y2 - y1
    ratio  = width / height if height > 0 else 0.0

    for label, lo, hi in _RATIO_RULES:
        if lo <= ratio < hi:
            return label

    return "4_seater_plus"


def _draw_annotated(image_bgr, detections, sofa_type, request_folder, ext=".jpg"):
    """Draw bounding boxes + label on a copy of the image and save it."""
    annotated = image_bgr.copy()
    colour = (0, 200, 0) if sofa_type == SUPPORTED_TYPE else (0, 60, 220)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf  = det["confidence"]
        label = f"{_TYPE_LABELS.get(sofa_type, sofa_type)}  {conf:.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    save_path = os.path.join(request_folder, f"sofa_annotated{ext}")
    cv2.imwrite(save_path, annotated)
    return save_path


def _save_analysis(analysis, request_folder):
    """Write sofa_analysis.json into request_folder."""
    os.makedirs(request_folder, exist_ok=True)
    path = os.path.join(request_folder, "sofa_analysis.json")
    with open(path, "w") as f:
        json.dump(analysis, f, indent=4)
    analysis["analysis_path"] = path


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def validate_sofa(image_path, request_folder, detector=None):
    """
    Run Phase 3 sofa validation.

    Parameters
    ----------
    image_path     : str   – path to the input image
    request_folder : str   – folder where outputs (JSON, annotated image) are saved
    detector       : SofaDetectorBase | None
                     Detection backend to use.
                     Defaults to YOLOv8DetectorBackend (yolov8n.pt).
                     Pass a CustomSofaDetectorPlaceholder (or your own subclass)
                     to use a different model.

    Returns
    -------
    dict with validation results (also written to sofa_analysis.json).

    Raises
    ------
    FileNotFoundError   – image_path does not exist
    SofaValidationError – no sofa detected, or sofa type not supported
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image for annotation
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise SofaValidationError(f"Could not read image: {image_path}")

    img_h, img_w = image_bgr.shape[:2]
    ext       = os.path.splitext(image_path)[1].lower() or ".jpg"
    write_ext = ext if ext in (".jpg", ".jpeg", ".png") else ".jpg"

    # Use default YOLOv8 backend if none provided
    if detector is None:
        detector = YOLOv8DetectorBackend()

    # Run detection
    detections = detector.detect(image_path)

    # ── Gate 1: no sofa at all ──────────────────────────────────────────────
    if not detections:
        analysis = {
            "validation_passed":    False,
            "error":                "Not a valid image. No sofa detected.",
            "detected_object":      None,
            "predicted_type":       None,
            "confidence":           None,
            "image_width":          img_w,
            "image_height":         img_h,
            "bbox":                 None,
            "aspect_ratio":         None,
            "annotated_image_path": None,
            "detector_backend":     type(detector).__name__,
        }
        _save_analysis(analysis, request_folder)
        raise SofaValidationError(analysis["error"])

    # ── Classify sofa type ──────────────────────────────────────────────────
    sofa_type = _classify_type(detections)
    best      = detections[0]
    x1, y1, x2, y2 = best["bbox"]
    aspect_ratio = round((x2 - x1) / (y2 - y1), 4) if (y2 - y1) > 0 else None

    # Draw annotated image
    annotated_path = _draw_annotated(image_bgr, detections[:2], sofa_type,
                                     request_folder, write_ext)

    # ── Gate 2: unsupported sofa type ───────────────────────────────────────
    if sofa_type != SUPPORTED_TYPE:
        human_label = _TYPE_LABELS.get(sofa_type, sofa_type)
        error_msg   = (
            f"Dimensions for {human_label} not available. "
            f"Only 3-seater sofa is currently supported."
        )
        analysis = {
            "validation_passed":    False,
            "error":                error_msg,
            "detected_object":      "sofa",
            "predicted_type":       sofa_type,
            "confidence":           best["confidence"],
            "image_width":          img_w,
            "image_height":         img_h,
            "bbox":                 best["bbox"],
            "aspect_ratio":         aspect_ratio,
            "annotated_image_path": annotated_path,
            "detector_backend":     type(detector).__name__,
        }
        _save_analysis(analysis, request_folder)
        raise SofaValidationError(error_msg)

    # ── All clear: 3-seater ─────────────────────────────────────────────────
    analysis = {
        "validation_passed":    True,
        "detected_object":      "sofa",
        "predicted_type":       sofa_type,
        "confidence":           best["confidence"],
        "image_width":          img_w,
        "image_height":         img_h,
        "bbox":                 best["bbox"],
        "aspect_ratio":         aspect_ratio,
        "annotated_image_path": annotated_path,
        "detector_backend":     type(detector).__name__,
    }
    _save_analysis(analysis, request_folder)
    return analysis
