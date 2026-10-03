"""
offline.py — Stallion runs with no external APIs or network calls.

Import this module BEFORE anything imports `ultralytics`. It:
  * sets YOLO_OFFLINE=1 so Ultralytics skips its internet probe, update
    checks and downloads (the flag is read once, at import time)
  * switches off Ultralytics' anonymous usage analytics ("sync") and its
    HUB / experiment-tracker integrations, which are on by default

All models, weights and viewer assets are local files in this repository.
"""

import os

os.environ["YOLO_OFFLINE"] = "1"
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("COMET_MODE", "disabled")

_NETWORK_SETTINGS = ("sync", "hub", "clearml", "comet", "dvc", "mlflow",
                     "neptune", "raytune", "tensorboard", "wandb")


def disable_ultralytics_network():
    """Persistently turn off Ultralytics analytics and integrations."""
    try:
        from ultralytics import settings
    except Exception:
        return
    off = {k: False for k in _NETWORK_SETTINGS if settings.get(k)}
    if off:
        settings.update(off)


disable_ultralytics_network()
