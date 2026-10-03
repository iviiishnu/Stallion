"""
eval_model.py — Evaluate the trained Stallion sofa segmentation model.

Runs inference on the validation set, computes per-class metrics,
saves a confusion matrix image, and writes a CSV + JSON report.

Run:
    python eval_model.py
    python eval_model.py --weights models/sofa_detector.pt --conf 0.25
"""

import offline  # noqa: F401 — no network: must run before ultralytics is imported
import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

SRC_DIR      = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent.resolve()
MODELS_DIR   = PROJECT_ROOT / "models"
EVAL_DIR     = PROJECT_ROOT / "outputs" / "eval"
DATA_YAML    = PROJECT_ROOT / "data" / "3seater_data" / "data.yaml"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "back_cushion", "base", "left_arm", "legs",
    "right_arm", "seat_cushion", "stallion-sofa"
]

# Minimum acceptable mAP@50 before a warning is shown
MAP50_THRESHOLD = 0.70


def evaluate(weights_path: Path, conf: float, iou: float):
    from ultralytics import YOLO

    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        print("  → Run train_model.py first.")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml not found: {DATA_YAML}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Stallion Sofa — Model Evaluation")
    print(f"  Weights:  {weights_path}")
    print(f"  Conf:     {conf}   IoU: {iou}")
    print(f"{'='*60}\n")

    model = YOLO(str(weights_path))

    # Run val on the validation split
    metrics = model.val(
        data    = str(DATA_YAML),
        conf    = conf,
        iou     = iou,
        plots   = True,
        save_json = True,
        project = str(EVAL_DIR),
        name    = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        verbose = True,
    )

    # ── Extract per-class metrics ─────────────────────────────────────────────
    results_dict = metrics.results_dict

    # Build per-class rows from the metrics object
    per_class_rows = []
    try:
        ap_class = metrics.box.ap_class_index   # class indices that have AP
        ap50     = metrics.box.ap50              # mAP50 per class
        prec     = metrics.box.p                 # precision per class
        rec      = metrics.box.r                 # recall per class

        for i, cls_idx in enumerate(ap_class):
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
            per_class_rows.append({
                "class":     cls_name,
                "precision": round(float(prec[i]), 4),
                "recall":    round(float(rec[i]),  4),
                "ap50":      round(float(ap50[i]), 4),
                "f1":        round(2 * float(prec[i]) * float(rec[i]) /
                                   max(float(prec[i]) + float(rec[i]), 1e-9), 4),
            })
    except AttributeError:
        print("[!] Could not extract per-class metrics — saving overall only.")

    # Overall metrics
    overall = {
        "map50_box":  round(float(results_dict.get("metrics/mAP50(B)",  0)), 4),
        "map50_seg":  round(float(results_dict.get("metrics/mAP50(M)",  0)), 4),
        "precision":  round(float(results_dict.get("metrics/precision(B)", 0)), 4),
        "recall":     round(float(results_dict.get("metrics/recall(B)",    0)), 4),
    }
    overall["f1"] = round(
        2 * overall["precision"] * overall["recall"] /
        max(overall["precision"] + overall["recall"], 1e-9), 4
    )

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n── Overall Metrics ──────────────────────────────────────")
    print(f"  mAP@50 (box):  {overall['map50_box']:.3f}")
    print(f"  mAP@50 (seg):  {overall['map50_seg']:.3f}")
    print(f"  Precision:     {overall['precision']:.3f}")
    print(f"  Recall:        {overall['recall']:.3f}")
    print(f"  F1:            {overall['f1']:.3f}")

    if per_class_rows:
        print("\n── Per-Class Metrics ────────────────────────────────────")
        header = f"  {'Class':<20} {'Precision':>10} {'Recall':>8} {'AP@50':>8} {'F1':>8}"
        print(header)
        print("  " + "-" * 58)
        for row in per_class_rows:
            print(f"  {row['class']:<20} {row['precision']:>10.3f} "
                  f"{row['recall']:>8.3f} {row['ap50']:>8.3f} {row['f1']:>8.3f}")

    # ── Accuracy gate ─────────────────────────────────────────────────────────
    print()
    if overall["map50_box"] >= MAP50_THRESHOLD:
        print(f"[✓] PASS — mAP@50 {overall['map50_box']:.3f} ≥ threshold {MAP50_THRESHOLD}")
    else:
        print(f"[✗] WARN — mAP@50 {overall['map50_box']:.3f} < threshold {MAP50_THRESHOLD}")
        print("     Consider more data, longer training, or data augmentation.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = EVAL_DIR / f"eval_{timestamp}_per_class.csv"
    if per_class_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "ap50", "f1"])
            writer.writeheader()
            writer.writerows(per_class_rows)
        print(f"[✓] Per-class CSV → {csv_path}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    json_path = EVAL_DIR / f"eval_{timestamp}_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "weights":    str(weights_path),
            "conf":       conf,
            "iou":        iou,
            "overall":    overall,
            "per_class":  per_class_rows,
            "pass":       overall["map50_box"] >= MAP50_THRESHOLD,
        }, f, indent=4)
    print(f"[✓] Summary JSON → {json_path}\n")

    return overall, per_class_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stallion sofa model")
    parser.add_argument("--weights", type=str,   default=str(MODELS_DIR / "sofa_detector.pt"))
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.50)
    args = parser.parse_args()

    evaluate(
        weights_path = Path(args.weights),
        conf         = args.conf,
        iou          = args.iou,
    )
