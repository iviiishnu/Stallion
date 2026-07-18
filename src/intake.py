"""
Phase 2 - Request intake CLI.

Basic (non-GUI) interface to get a request from a customer into the
Phase 1 pipeline: takes image + dimensions (as CLI args, or prompts for
whatever's missing), validates/processes the image, then runs the
existing cost engine end-to-end.

Usage:
    python intake.py --image path/to/sofa.jpg --length 2400 --width 950 --height 900
    python intake.py                      # prompts for everything interactively
"""

import os
import sys
import argparse
import json

from image_processor import process_image, ImageValidationError
from sofa_validator import validate_sofa, SofaValidationError
from run_quote import (
    validate_request,
    create_request_folder,
    copy_generated_outputs,
    save_request_summary,
)
from cost_engine import SofaCostEngine


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def next_request_id(project_root):
    """Auto-increments demo_001, demo_002, ... (same scheme request_form.py used)."""
    requests_folder = os.path.join(project_root, "outputs", "requests")
    os.makedirs(requests_folder, exist_ok=True)
    count = 1
    while os.path.exists(os.path.join(requests_folder, f"demo_{count:03d}")):
        count += 1
    return f"demo_{count:03d}"


def prompt_for(label, cast=str, validator=None):
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = cast(raw)
        except ValueError:
            print(f"  Invalid value, expected a {cast.__name__}. Try again.")
            continue
        if validator and not validator(value):
            print("  Invalid value. Try again.")
            continue
        return value


def gather_request(args):
    """Fills in any missing fields via interactive prompts."""
    image_path = args.image
    if image_path and not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        image_path = None
    if not image_path:
        image_path = prompt_for(
            "Sofa image path", validator=lambda p: os.path.exists(p)
        )
    length = args.length if args.length is not None else prompt_for(
        "Length (mm)", cast=float, validator=lambda v: v > 0
    )
    width = args.width if args.width is not None else prompt_for(
        "Width (mm)", cast=float, validator=lambda v: v > 0
    )
    height = args.height if args.height is not None else prompt_for(
        "Height (mm)", cast=float, validator=lambda v: v > 0
    )
    customer_name = args.customer_name or input("Customer name [guest]: ").strip() or "guest"
    sofa_type = args.sofa_type or "3_seater"
    request_id = args.request_id or next_request_id(PROJECT_ROOT)

    return {
        "request_id": request_id,
        "customer_name": customer_name,
        "sofa_type": sofa_type,
        "image_path": os.path.abspath(image_path),
        "dimensions_mm": {"length": length, "width": width, "height": height},
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 request intake")
    parser.add_argument("--image", help="Path to sofa image")
    parser.add_argument("--length", type=float, help="Length in mm")
    parser.add_argument("--width", type=float, help="Width in mm")
    parser.add_argument("--height", type=float, help="Height in mm")
    parser.add_argument("--customer_name")
    parser.add_argument("--sofa_type", default=None)
    parser.add_argument("--request_id", default=None, help="Defaults to auto-generated demo_NNN")
    args = parser.parse_args()

    request_data = gather_request(args)
    request_id = request_data["request_id"]

    print("\n===== REQUEST INTAKE =====")
    print(json.dumps(request_data, indent=4))

    # Validate request shape + resolve image path (reuses Phase 1's validation)
    try:
        image_abs_path = validate_request(request_data, PROJECT_ROOT)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[FAILED] Invalid request: {exc}")
        sys.exit(1)

    # Create the request's output folder up front, since both image
    # processing and quote generation write into it.
    request_folder = create_request_folder(PROJECT_ROOT, request_id)

    # --- Phase 2: image intake/validation ---
    try:
        image_result = process_image(image_abs_path, request_folder)
    except (FileNotFoundError, ImageValidationError) as exc:
        print(f"\n[FAILED] Image processing error: {exc}")
        sys.exit(1)

    print("\n===== IMAGE PROCESSED =====")
    print(json.dumps(image_result["metadata"], indent=4))

    # --- Phase 3: sofa type validation (hard gate) ---
    print("\n===== SOFA VALIDATION =====")
    try:
        sofa_result = validate_sofa(image_abs_path, request_folder)
    except SofaValidationError as exc:
        print(f"\n[REJECTED] {exc}")
        sys.exit(1)

    print(f"  Detected  : {sofa_result['detected_object']}")
    print(f"  Type      : {sofa_result['predicted_type']}")
    print(f"  Confidence: {sofa_result['confidence']:.0%}")
    print(f"  BBox      : {sofa_result['bbox']}")
    print(f"  Aspect ratio: {sofa_result['aspect_ratio']}")
    print("  ✅ 3-seater confirmed — proceeding to cost engine")

    # --- Phase 1: cost engine (unchanged) ---
    dims = request_data["dimensions_mm"]
    engine = SofaCostEngine()
    quote_result = engine.generate_quote(
        length_mm=dims["length"],
        width_mm=dims["width"],
        height_mm=dims["height"],
        output_prefix=request_id,
    )

    # Write input_request.json directly into the request folder (single copy, no temp file).
    input_request_path = os.path.join(request_folder, "input_request.json")
    with open(input_request_path, "w") as f:
        json.dump(request_data, f, indent=4)

    copied_paths = copy_generated_outputs(quote_result, request_folder)
    copied_paths.update(
        {
            "input_request":    input_request_path,
            "original_image":   image_result["original_image_path"],
            "processed_image":  image_result["processed_image_path"],
            "image_metadata":   image_result["metadata_path"],
            "sofa_analysis":    sofa_result.get("analysis_path", ""),
            "sofa_annotated":   sofa_result.get("annotated_image_path", ""),
        }
    )

    summary_path = save_request_summary(
        request_folder=request_folder,
        request_data=request_data,
        image_abs_path=image_abs_path,
        quote_result=quote_result,
        copied_paths=copied_paths,
    )

    print("\n===== QUOTATION GENERATED =====")
    print(json.dumps(quote_result["summary"] if "summary" in quote_result else quote_result, indent=4))
    print(f"\nRequest summary saved to: {summary_path}")
    print(f"Bundled request folder: {request_folder}")


if __name__ == "__main__":
    main()
