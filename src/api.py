"""
FastAPI backend for the Stallion Sofa Costing Engine.

Wraps the existing Phase 1-3 pipeline as REST endpoints so the
web frontend can call it without touching any pipeline code.

Endpoints:
  GET  /              → serve frontend/index.html
  POST /api/quote     → run the full pipeline, return JSON
  GET  /outputs/...   → serve output files (annotated image, etc.)
  GET  /static/...    → serve frontend assets

Run:
  python -m uvicorn api:app --reload --port 8000
"""

import os
import sys
import csv
import json
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Path setup — api.py lives in src/, pipeline modules are siblings
# ---------------------------------------------------------------------------

SRC_DIR     = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent.resolve()
FRONTEND_DIR = PROJECT_ROOT / "frontend"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(SRC_DIR))

from image_processor import process_image, ImageValidationError
from sofa_validator   import validate_sofa, SofaValidationError
from cost_engine      import SofaCostEngine

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Stallion Sofa Costing Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output files (annotated images, CSVs, etc.)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Serve frontend static assets (CSS, JS, icons)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helper: read a CSV file into a list of dicts, coercing numeric strings
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            clean = {}
            for k, v in row.items():
                if k is None:
                    continue
                try:
                    clean[k] = float(v) if "." in v else int(v)
                except (ValueError, TypeError):
                    clean[k] = v
            rows.append(clean)
    return rows


def _next_request_id() -> str:
    """web_YYYYMMDD_HHMMSS — unique per second, human-readable."""
    return f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_index():
    """Serve the single-page frontend."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/quote")
async def create_quote(
    image:         UploadFile = File(...),
    customer_name: str   = Form("guest"),
    length_mm:     float = Form(...),
    width_mm:      float = Form(...),
    height_mm:     float = Form(...),
):
    """
    Run the Phase 1-3 pipeline on the uploaded image + dimensions.

    Returns a JSON response with:
      status          "success" | "rejected" | "error"
      sofa_analysis   detection results (always present if image was valid)
      bom             list of scaled component rows
      quote           cost breakdown dict
      annotated_image_url   URL to the bbox-annotated image
    """
    request_id     = _next_request_id()
    request_folder = OUTPUTS_DIR / "requests" / request_id
    request_folder.mkdir(parents=True, exist_ok=True)

    # ── Save uploaded image ──────────────────────────────────────────────────
    suffix     = Path(image.filename or "upload.jpg").suffix.lower() or ".jpg"
    image_path = request_folder / f"uploaded_image{suffix}"
    content    = await image.read()
    with open(image_path, "wb") as f:
        f.write(content)

    # ── Phase 2: image validation & processing ───────────────────────────────
    try:
        image_result = process_image(str(image_path), str(request_folder))
    except (FileNotFoundError, ImageValidationError) as exc:
        return JSONResponse(
            {"status": "error", "reason": str(exc)},
            status_code=400,
        )

    # ── Phase 3: sofa type validation (hard gate) ────────────────────────────
    sofa_result   = {}
    annotated_url = None

    try:
        sofa_result = validate_sofa(str(image_path), str(request_folder))
        # Convert absolute annotated path → URL
        ann_path = sofa_result.get("annotated_image_path", "")
        if ann_path and Path(ann_path).exists():
            rel = Path(ann_path).relative_to(OUTPUTS_DIR)
            annotated_url = f"/outputs/{rel.as_posix()}"
        # Remove non-serialisable absolute path from response
        sofa_result.pop("annotated_image_path", None)
        sofa_result.pop("analysis_path", None)

    except SofaValidationError as exc:
        # Read sofa_analysis.json for type/confidence details
        analysis_json = request_folder / "sofa_analysis.json"
        if analysis_json.exists():
            with open(analysis_json) as f:
                sofa_result = json.load(f)
        sofa_result.pop("annotated_image_path", None)
        sofa_result.pop("analysis_path", None)

        ann_path = request_folder / "sofa_annotated.jpg"
        if ann_path.exists():
            annotated_url = f"/outputs/requests/{request_id}/sofa_annotated.jpg"

        return JSONResponse({
            "status":         "rejected",
            "reason":         str(exc),
            "request_id":     request_id,
            "sofa_analysis":  sofa_result,
            "annotated_image_url": annotated_url,
        })

    # ── Phase 1: cost engine ─────────────────────────────────────────────────
    engine      = SofaCostEngine()
    quote_result = engine.generate_quote(
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        output_prefix=request_id,
    )

    output_files = quote_result.get("output_files", {})

    # Copy BOM + cost CSVs into request folder and read them back as JSON
    bom_rows  = []
    cost_rows = []

    if output_files.get("bom_csv") and Path(output_files["bom_csv"]).exists():
        dst = request_folder / "bom.csv"
        shutil.copy2(output_files["bom_csv"], dst)
        bom_rows = _read_csv(dst)

    if output_files.get("cost_csv") and Path(output_files["cost_csv"]).exists():
        dst = request_folder / "cost.csv"
        shutil.copy2(output_files["cost_csv"], dst)
        cost_rows = _read_csv(dst)

    if output_files.get("summary_json") and Path(output_files["summary_json"]).exists():
        shutil.copy2(output_files["summary_json"], request_folder / "quote_summary.json")

    # Merge BOM scaling info into cost rows (same order, join on component_group)
    bom_by_name = {r.get("component_group", ""): r for r in bom_rows}
    merged_bom  = []
    for row in cost_rows:
        name = row.get("component_group", "")
        bom_info = bom_by_name.get(name, {})
        merged_bom.append({
            "component":    name,
            "scaling_rule": bom_info.get("scaling_rule", "—"),
            "qty":          round(float(row.get("new_qty", 0)), 3),
            "unit_cost":    float(row.get("unit_cost", 0)),
            "total_cost":   round(float(row.get("total_cost", 0)), 2),
        })

    # quote summary
    summary = quote_result.get("summary", quote_result)

    # Save input request JSON
    with open(request_folder / "input_request.json", "w") as f:
        json.dump({
            "request_id":    request_id,
            "customer_name": customer_name,
            "sofa_type":     "3_seater",
            "image_path":    str(image_path),
            "dimensions_mm": {"length": length_mm, "width": width_mm, "height": height_mm},
        }, f, indent=4)

    return JSONResponse({
        "status":       "success",
        "request_id":   request_id,
        "customer_name": customer_name,
        "dimensions_mm": {"length": length_mm, "width": width_mm, "height": height_mm},
        "image_metadata":    image_result["metadata"],
        "sofa_analysis":     sofa_result,
        "bom":               merged_bom,
        "quote":             summary,
        "annotated_image_url": annotated_url,
    })
