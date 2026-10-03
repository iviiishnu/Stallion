"""
train_model.py — Fine-tune YOLOv8 segmentation on the Stallion 3-seater dataset.

Dataset:  data/3seater_data/
Classes:  back_cushion, base, left_arm, legs, right_arm, seat_cushion, stallion-sofa
Output:   models/sofa_detector/  (weights, metrics, plots)

Run:
    python train_model.py
    python train_model.py --epochs 100 --batch 8 --img 640
"""

import offline  # noqa: F401 — no network: must run before ultralytics is imported
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent.resolve()
DATA_YAML    = PROJECT_ROOT / "data" / "3seater_data" / "data.yaml"
MODELS_DIR   = PROJECT_ROOT / "models"
RUNS_DIR     = PROJECT_ROOT / "outputs" / "training_runs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def patch_data_yaml(data_yaml: Path) -> Path:
    """
    YOLOv8 needs absolute paths in data.yaml when called from a different CWD.
    We write a patched copy with absolute train/val paths so it works from anywhere.
    """
    import yaml  # pyyaml — installed with ultralytics

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    dataset_root = data_yaml.parent.resolve()
    cfg["path"] = str(dataset_root)
    cfg["train"] = "train/images"
    cfg["val"]   = "valid/images"

    patched_path = dataset_root / "_data_abs.yaml"
    with open(patched_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return patched_path


def train(epochs: int, batch: int, img_size: int, device: str, resume: bool):
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  Stallion Sofa — YOLOv8 Segmentation Training")
    print(f"  Epochs: {epochs}  |  Batch: {batch}  |  Img: {img_size}")
    print(f"  Dataset: {DATA_YAML}")
    print(f"{'='*60}\n")

    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml not found at: {DATA_YAML}")
        sys.exit(1)

    # Patch yaml to absolute paths
    abs_yaml = patch_data_yaml(DATA_YAML)

    # Run name with timestamp
    run_name = f"stallion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load base model (segmentation variant)
    model = YOLO("yolov8n-seg.pt")

    # Train
    results = model.train(
        data     = str(abs_yaml),
        epochs   = epochs,
        batch    = batch,
        imgsz    = img_size,
        device   = device,
        name     = run_name,
        project  = str(RUNS_DIR),
        patience = 20,           # early stopping
        save     = True,
        plots    = True,
        resume   = resume,
        # Augmentation — conservative for small dataset
        degrees  = 10.0,
        flipud   = 0.1,
        fliplr   = 0.5,
        mosaic   = 0.8,
        mixup    = 0.0,
        hsv_h    = 0.015,
        hsv_s    = 0.5,
        hsv_v    = 0.3,
    )

    # ── Copy best weights to models/ ─────────────────────────────────────────
    best_weights = RUNS_DIR / run_name / "weights" / "best.pt"
    if best_weights.exists():
        dst = MODELS_DIR / "sofa_detector.pt"
        shutil.copy2(best_weights, dst)
        print(f"\n[✓] Best weights saved → {dst}")
    else:
        print(f"\n[!] best.pt not found at {best_weights}")

    # ── Save training summary JSON ────────────────────────────────────────────
    try:
        summary = {
            "run_name":     run_name,
            "epochs":       epochs,
            "batch":        batch,
            "img_size":     img_size,
            "best_weights": str(MODELS_DIR / "sofa_detector.pt"),
            "run_dir":      str(RUNS_DIR / run_name),
            "metrics": {
                "box_map50":  float(results.results_dict.get("metrics/mAP50(B)",  0)),
                "seg_map50":  float(results.results_dict.get("metrics/mAP50(M)",  0)),
                "precision":  float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall":     float(results.results_dict.get("metrics/recall(B)",    0)),
            },
        }
        summary_path = MODELS_DIR / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[✓] Training summary → {summary_path}")
        print(f"\n  mAP@50 (box): {summary['metrics']['box_map50']:.3f}")
        print(f"  mAP@50 (seg): {summary['metrics']['seg_map50']:.3f}")
        print(f"  Precision:    {summary['metrics']['precision']:.3f}")
        print(f"  Recall:       {summary['metrics']['recall']:.3f}")
    except Exception as e:
        print(f"[!] Could not save summary: {e}")

    print(f"\n[✓] Training complete. Run dir: {RUNS_DIR / run_name}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stallion sofa segmentation model")
    parser.add_argument("--epochs", type=int,   default=80,    help="Training epochs (default 80)")
    parser.add_argument("--batch",  type=int,   default=8,     help="Batch size (default 8)")
    parser.add_argument("--img",    type=int,   default=640,   help="Image size (default 640)")
    parser.add_argument("--device", type=str,   default="cpu", help="Device: cpu / 0 / cuda")
    parser.add_argument("--resume", action="store_true",       help="Resume from last checkpoint")
    args = parser.parse_args()

    train(
        epochs   = args.epochs,
        batch    = args.batch,
        img_size = args.img,
        device   = args.device,
        resume   = args.resume,
    )
