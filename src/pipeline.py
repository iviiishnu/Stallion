"""
pipeline.py — Ties the whole Stallion flow together.

image + L,W,H
      │
      ▼
YOLO segmentation / sofa_validator
      │
      ├── sofa type
      └── component masks
              │
              ▼
       shape_analysis.py
              │
              ▼
        design_params
              │
              ├──────────────┐
              ▼              ▼
        scaled BOM/cost    CAD geometry
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                  2D CAD              3D CAD

YOLO is intentionally executed only once.
The component segmentation produced by sofa_validator.py
is passed directly into shape_analysis.py.
"""

import offline  # noqa: F401 — no network: must run before ultralytics is imported
from sofa_validator import validate_sofa, SofaValidationError
from cost_engine import SofaCostEngine
from bom_bridge import bom_df_to_geometry_dict
from cad_generator import SofaGeometry, SofaCADGenerator
from sofa_3d_generator import Sofa3DGenerator
from shape_analysis import analyze_image


# ---------------------------------------------------------------------------
# Detector sofa labels -> cost engine labels
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "1_seater": "1-seater",
    "2_seater": "2-seater",
    "3_seater": "3-seater",
    "4_seater_plus": "4-seater",
    "l_shape": "l-shape",
}


class PipelineError(Exception):
    """Raised when any pipeline stage fails; wraps the original error."""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_full_quote(
    image_path: str,
    request_folder: str,
    length_mm: float,
    width_mm: float,
    height_mm: float,
    request_id: str,
    cad_output_dir: str,
    cost_engine_base_dir: str = "..",
    detector=None,
):
    """
    Run the complete Stallion pipeline.

    Stages:

        1. Sofa detection + component segmentation
        2. Shape analysis from segmentation masks
        3. BOM scaling + costing
        4. Shape-adapted 2D CAD
        5. Shape-adapted 3D CAD

    Returns a single dictionary containing detection results,
    shape parameters, quotation data and all generated file paths.
    """

    # ======================================================================
    # 1. DETECTION + SEGMENTATION
    # ======================================================================

    try:

        analysis = validate_sofa(
            image_path,
            request_folder,
            detector=detector,
        )

    except SofaValidationError:
        # Validation errors are intentionally allowed to propagate.
        raise

    except Exception as e:

        raise PipelineError(
            f"Sofa detection/segmentation failed: {e}"
        ) from e

    detected_type = analysis["predicted_type"]

    sofa_type = _TYPE_MAP.get(
        detected_type,
        "3-seater",
    )

    # The trained Stallion YOLO model already gives us component
    # segmentation. Reuse those results instead of running YOLO again.
    component_detections = analysis.get(
        "component_detections",
        [],
    )

    # ======================================================================
    # 2. SHAPE ANALYSIS
    # ======================================================================

    try:

        shape = analyze_image(
            image_path=image_path,
            component_detections=component_detections,
        )

    except Exception as e:

        # Shape analysis must never break the costing/CAD pipeline.
        # If it fails, the CAD system receives {} and uses its
        # calibrated template defaults.
        shape = {
            "design_params": {},
            "confidence": 0.0,
            "measurements": {},
            "source": "yolo_segmentation",
            "error": str(e),
        }

    design_params = shape.get(
        "design_params",
        {},
    )

    # ======================================================================
    # 3. COST ENGINE
    # ======================================================================

    try:

        engine = SofaCostEngine(
            base_dir=cost_engine_base_dir
        )

        engine.load_data()

        try:

            scales, bom_df = (
                engine.generate_scaled_bom(
                    length_mm,
                    width_mm,
                    height_mm,
                    sofa_type,
                )
            )

        except ValueError as e:

            raise ValueError(
                f"Dimension validation failed "
                f"for {sofa_type}: {e}"
            ) from e

        cost_df, cost_summary = (
            engine.compute_cost(
                bom_df
            )
        )

        (
            bom_path,
            quote_csv_path,
            quote_json_path,
            _,
        ) = engine.save_outputs(
            bom_df=bom_df,
            cost_df=cost_df,
            summary=cost_summary,
            output_prefix=request_id,
        )

    except ValueError:
        raise

    except Exception as e:

        raise PipelineError(
            f"Cost engine failed: {e}"
        ) from e

    # ======================================================================
    # 4. BOM -> GEOMETRY
    # ======================================================================

    try:

        geo_bom = bom_df_to_geometry_dict(
            bom_df
        )

    except Exception as e:

        raise PipelineError(
            f"BOM-to-geometry conversion failed: {e}"
        ) from e

    # ======================================================================
    # 5. SHAPE-ADAPTED GEOMETRY
    # ======================================================================

    try:

        geo = SofaGeometry(
            L=length_mm,
            W=width_mm,
            H=height_mm,
            bom=geo_bom,
            design_params=design_params,
        )

    except TypeError as e:

        raise PipelineError(
            "SofaGeometry does not yet accept "
            "'design_params'. Update cad_generator.py "
            "before running the full pipeline."
        ) from e

    except Exception as e:

        raise PipelineError(
            f"3D geometry initialization failed: {e}"
        ) from e

    # ======================================================================
    # 6. 2D CAD
    # ======================================================================

    try:

        cad2d = SofaCADGenerator(
            L=length_mm,
            W=width_mm,
            H=height_mm,
            bom=geo_bom,
            request_id=request_id,
            sofa_type=sofa_type,
            design_params=design_params,
        )

        png_path = cad2d.export_png(
            cad_output_dir
        )

        dxf_path = cad2d.export_dxf(
            cad_output_dir
        )

    except TypeError as e:

        raise PipelineError(
            "SofaCADGenerator does not yet accept "
            "'design_params'. Update cad_generator.py "
            "before running the full pipeline."
        ) from e

    except Exception as e:

        raise PipelineError(
            f"2D CAD generation failed: {e}"
        ) from e

    # ======================================================================
    # 7. 3D CAD
    # ======================================================================

    try:

        cad3d = Sofa3DGenerator(
            geo,
            request_id=request_id,
            sofa_type=sofa_type,
        )

        model_files = cad3d.export_all(
            cad_output_dir
        )

        preview_external = (
            cad3d.export_preview_png(
                cad_output_dir,
                exploded=False,
                groups=Sofa3DGenerator.EXTERNAL_GROUPS,
                label="preview_shell",
            )
        )

        preview_internal = (
            cad3d.export_preview_png(
                cad_output_dir,
                exploded=False,
                groups=Sofa3DGenerator.INTERNAL_GROUPS,
                label="preview_structure",
            )
        )

    except Exception as e:

        raise PipelineError(
            f"3D CAD generation failed: {e}"
        ) from e

    # ======================================================================
    # 8. FINAL RESULT
    # ======================================================================

    return {

        # Original detector output
        "detection": analysis,

        # Shape analysis output
        "shape_analysis": shape,

        # Convenient direct access
        "design_params": design_params,

        # Detected sofa type
        "sofa_type": sofa_type,

        # Cost engine scaling
        "scale_factors": {
            "SL": scales["SL"],
            "SW": scales["SW"],
            "SH": scales["SH"],
        },

        # Cost / quotation
        "cost_summary": cost_summary,

        # All generated files
        "output_files": {

            "annotated_image":
                analysis[
                    "annotated_image_path"
                ],

            "bom_csv":
                bom_path,

            "cost_csv":
                quote_csv_path,

            "quotation_json":
                quote_json_path,

            "cad_sheet_png":
                png_path,

            "cad_dxf":
                dxf_path,

            "model_glb":
                model_files["glb"],

            "model_glb_exploded":
                model_files["glb_exploded"],

            "model_stl":
                model_files["stl"],

            "model_step":
                model_files["step"],

            "model_takeoff_csv":
                model_files["takeoff_csv"],

            "model_takeoff_json":
                model_files["takeoff_json"],

            "model_preview_shell_png":
                preview_external,

            "model_preview_structure_png":
                preview_internal,
        },
    }