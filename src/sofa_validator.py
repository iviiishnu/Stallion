"""
Stallion Sofa Validator + YOLO Segmentation

Uses the trained Stallion YOLO segmentation model:

    ../models/Sofa-YOLO-Training-main/models/best.pt

The model classes are:

    0: back_cushion
    1: base
    2: left_arm
    3: legs
    4: right_arm
    5: seat_cushion
    6: one-seater
    7: stallion-sofa1
    8: stallion-sofa2
    9: stallion-sofa
    10: two-seater

The important component masks are preserved and passed through
the validation result so shape_analysis.py can use them.
"""

import offline  # noqa: F401 — no network: must run before ultralytics is imported
import os
import json
from abc import ABC, abstractmethod
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CONFIDENCE = 0.25

SUPPORTED_TYPES = [
    "1_seater",
    "2_seater",
    "3_seater",
    "4_seater_plus",
    "l_shape",
]

_TYPE_LABELS = {
    "1_seater": "1-seater (armchair)",
    "2_seater": "2-seater (loveseat)",
    "3_seater": "3-seater",
    "4_seater_plus": "4-seater / large sofa",
    "l_shape": "L-shape / sectional",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SofaValidationError(ValueError):
    """Raised when an image fails sofa validation."""


# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------

class SofaDetectorBase(ABC):

    @abstractmethod
    def detect(self, image_path: str) -> list:
        """
        Return detections.

        Each detection contains at minimum:

            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
            }

        Custom segmentation detections may additionally contain:

            "class_id"
            "class_name"
            "mask"
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CUSTOM STALLION YOLO SEGMENTATION MODEL
# ---------------------------------------------------------------------------

class StallionSofaDetector(SofaDetectorBase):
    """
    Uses the actual trained Stallion YOLO segmentation model.

    Model:
        models/Sofa-YOLO-Training-main/models/best.pt

    Component classes:
        back_cushion
        base
        left_arm
        legs
        right_arm
        seat_cushion

    Sofa/type classes:
        one-seater
        two-seater
        stallion-sofa1
        stallion-sofa2
        stallion-sofa

    All component segmentation masks are preserved.
    """

    COMPONENT_CLASSES = {
        "back_cushion",
        "base",
        "left_arm",
        "legs",
        "right_arm",
        "seat_cushion",
    }

    SOFA_CLASSES = {
        "one-seater",
        "two-seater",
        "stallion-sofa1",
        "stallion-sofa2",
        "stallion-sofa",
    }

    def __init__(self, weights_path: str = None):

        from ultralytics import YOLO

        src_dir = Path(__file__).parent.resolve()
        proj_root = src_dir.parent.resolve()

        if weights_path:
            wp = Path(weights_path)
        else:
            wp = (
                proj_root
                / "models"
                / "Sofa-YOLO-Training-main"
                / "models"
                / "best.pt"
            )

        if not wp.exists():
            raise FileNotFoundError(
                f"Trained Stallion segmentation model not found:\n{wp}"
            )

        print(f"[detector] Loading Stallion segmentation model:")
        print(f"[detector] {wp}")

        self._model = YOLO(str(wp), verbose=False)
        self.weights_path = str(wp)

        print(f"[detector] Model classes: {self._model.names}")

    def detect(self, image_path: str) -> list:

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        result = self._model(
            image_path,
            verbose=False
        )[0]

        detections = []

        if result.boxes is None:
            return detections

        for i, box in enumerate(result.boxes):

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < MIN_CONFIDENCE:
                continue

            class_name = self._model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = {
                "bbox": [
                    round(x1),
                    round(y1),
                    round(x2),
                    round(y2),
                ],
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "class_name": class_name,
            }

            # -----------------------------------------------------------
            # Preserve segmentation mask
            # -----------------------------------------------------------

            if result.masks is not None and i < len(result.masks):

                mask = result.masks.data[i]

                # Convert tensor → numpy
                mask = mask.cpu().numpy()

                # Store as a compact binary mask.
                # JSON cannot serialize numpy arrays, so convert to list.
                detection["mask"] = (
                    (mask > 0.5)
                    .astype("uint8")
                    .tolist()
                )

            detections.append(detection)

        # Highest confidence first
        detections.sort(
            key=lambda d: d["confidence"],
            reverse=True
        )

        return detections


# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

def get_detector(custom_weights: str = None) -> SofaDetectorBase:
    """
    Always prefer the trained Stallion segmentation model.

    If custom_weights is supplied, use that model.
    Otherwise automatically use:

        models/Sofa-YOLO-Training-main/models/best.pt
    """

    detector = StallionSofaDetector(
        weights_path=custom_weights
    )

    print(
        f"[detector] Using Stallion YOLO segmentation model: "
        f"{detector.weights_path}"
    )

    return detector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou(box_a, box_b):

    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    if inter == 0:
        return 0.0

    area_a = (
        (box_a[2] - box_a[0])
        * (box_a[3] - box_a[1])
    )

    area_b = (
        (box_b[2] - box_b[0])
        * (box_b[3] - box_b[1])
    )

    return inter / (area_a + area_b - inter)


def _component_detections(detections):

    return [
        d for d in detections
        if d.get("class_name") in StallionSofaDetector.COMPONENT_CLASSES
    ]


def _sofa_detections(detections):

    return [
        d for d in detections
        if d.get("class_name") in StallionSofaDetector.SOFA_CLASSES
    ]


def _combined_component_bbox(detections):

    components = _component_detections(detections)

    if not components:
        return None

    x1 = min(d["bbox"][0] for d in components)
    y1 = min(d["bbox"][1] for d in components)
    x2 = max(d["bbox"][2] for d in components)
    y2 = max(d["bbox"][3] for d in components)

    return [x1, y1, x2, y2]


def _classify_type(detections):

    # If model gives an explicit sofa-type class, use it.
    sofa_dets = _sofa_detections(detections)

    if sofa_dets:

        best = max(
            sofa_dets,
            key=lambda d: d["confidence"]
        )

        name = best["class_name"]

        if name == "one-seater":
            return "1_seater"

        if name == "two-seater":
            return "2_seater"

        # Current project is primarily 3-seater.
        if name in {
            "stallion-sofa",
            "stallion-sofa1",
            "stallion-sofa2",
        }:
            return "3_seater"

    # Fallback: determine from combined component bbox.
    bbox = _combined_component_bbox(detections)

    if bbox is None:
        return "3_seater"

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    ratio = width / height if height > 0 else 0

    if ratio < 1.30:
        return "1_seater"

    if ratio < 1.60:
        return "2_seater"

    if ratio < 3.20:
        return "3_seater"

    return "4_seater_plus"


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _draw_annotated(
    image_bgr,
    detections,
    sofa_type,
    request_folder,
    ext=".jpg",
):

    annotated = image_bgr.copy()

    # Different colours for different component types.
    colors = {
        "back_cushion": (255, 120, 0),
        "base": (120, 120, 120),
        "left_arm": (0, 200, 255),
        "legs": (100, 255, 100),
        "right_arm": (255, 0, 200),
        "seat_cushion": (0, 150, 255),
    }

    for det in detections:

        x1, y1, x2, y2 = [
            int(v) for v in det["bbox"]
        ]

        conf = det["confidence"]
        name = det.get("class_name", "sofa")

        color = colors.get(
            name,
            (0, 200, 0)
        )

        label = f"{name} {conf:.0%}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    save_path = os.path.join(
        request_folder,
        f"sofa_annotated{ext}"
    )

    os.makedirs(
        request_folder,
        exist_ok=True
    )

    cv2.imwrite(
        save_path,
        annotated
    )

    return save_path


# ---------------------------------------------------------------------------
# Analysis JSON
# ---------------------------------------------------------------------------

def _save_analysis(analysis, request_folder):

    os.makedirs(
        request_folder,
        exist_ok=True
    )

    path = os.path.join(
        request_folder,
        "sofa_analysis.json"
    )

    # Masks can make this JSON large.
    # Keep a separate lightweight representation for now.
    serializable = dict(analysis)

    with open(path, "w") as f:
        json.dump(
            serializable,
            f,
            indent=4
        )

    analysis["analysis_path"] = path

    return path


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def validate_sofa(
    image_path,
    request_folder,
    detector=None,
):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:

        raise SofaValidationError(
            f"Could not read image: {image_path}"
        )

    img_h, img_w = image_bgr.shape[:2]

    ext = (
        os.path.splitext(image_path)[1]
        .lower()
    )

    write_ext = (
        ext
        if ext in (".jpg", ".jpeg", ".png")
        else ".jpg"
    )

    # ---------------------------------------------------------------
    # Detector
    # ---------------------------------------------------------------

    if detector is None:
        detector = get_detector()

    detections = detector.detect(
        image_path
    )

    # ---------------------------------------------------------------
    # No detections
    # ---------------------------------------------------------------

    if not detections:

        analysis = {
            "validation_passed": False,
            "error": "No sofa/components detected.",
            "detected_object": None,
            "predicted_type": None,
            "confidence": None,
            "image_width": img_w,
            "image_height": img_h,
            "bbox": None,
            "aspect_ratio": None,
            "detections": [],
            "component_detections": [],
            "detector_backend": type(detector).__name__,
        }

        _save_analysis(
            analysis,
            request_folder
        )

        raise SofaValidationError(
            analysis["error"]
        )

    # ---------------------------------------------------------------
    # Sofa type
    # ---------------------------------------------------------------

    sofa_type = _classify_type(
        detections
    )

    # Prefer whole-sofa detection bbox.
    sofa_dets = _sofa_detections(
        detections
    )

    if sofa_dets:

        best = max(
            sofa_dets,
            key=lambda d: d["confidence"]
        )

    else:

        best = detections[0]

    x1, y1, x2, y2 = best["bbox"]

    aspect_ratio = (
        round(
            (x2 - x1) / (y2 - y1),
            4
        )
        if (y2 - y1) > 0
        else None
    )

    # ---------------------------------------------------------------
    # Components
    # ---------------------------------------------------------------

    component_detections = _component_detections(
        detections
    )

    component_counts = {}

    for det in component_detections:

        name = det["class_name"]

        component_counts[name] = (
            component_counts.get(name, 0)
            + 1
        )

    # ---------------------------------------------------------------
    # Annotated image
    # ---------------------------------------------------------------

    annotated_path = _draw_annotated(
        image_bgr,
        detections,
        sofa_type,
        request_folder,
        write_ext,
    )

    # ---------------------------------------------------------------
    # Final analysis
    # ---------------------------------------------------------------

    analysis = {
        "validation_passed": True,

        "detected_object": "sofa",

        "predicted_type": sofa_type,

        "confidence": best["confidence"],

        "image_width": img_w,

        "image_height": img_h,

        "bbox": best["bbox"],

        "aspect_ratio": aspect_ratio,

        "annotated_image_path": annotated_path,

        "detector_backend": type(detector).__name__,

        "model_weights": getattr(
            detector,
            "weights_path",
            None
        ),

        # ALL detections, including segmentation masks.
        "detections": detections,

        # Easier for shape_analysis.py to consume.
        "component_detections": component_detections,

        "component_counts": component_counts,

        "segmentation_available": any(
            "mask" in d
            for d in component_detections
        ),
    }

    _save_analysis(
        analysis,
        request_folder
    )

    return analysis