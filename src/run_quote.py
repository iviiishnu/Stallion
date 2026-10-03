import os
import json
import shutil
import argparse

from cost_engine import SofaCostEngine


# ---------------------------------------------------
# 1. LOAD REQUEST JSON
# ---------------------------------------------------
def load_request_json(request_json_path):
    with open(request_json_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------
# 2. VALIDATE REQUEST
# ---------------------------------------------------
def validate_request(request_data, project_root):
    """
    Expected request JSON format:
    {
        "request_id": "demo_001",
        "customer_name": "test_customer",
        "sofa_type": "3_seater",
        "image_path": "data/sample_inputs/sample_3seater_01.jpg",
        "dimensions_mm": {
            "length": 2400,
            "width": 950,
            "height": 900
        }
    }
    """

    required_top_keys = [
        "request_id",
        "customer_name",
        "sofa_type",
        "image_path",
        "dimensions_mm"
    ]

    for key in required_top_keys:
        if key not in request_data:
            raise ValueError(f"Missing required field in request JSON: '{key}'")

    dims = request_data["dimensions_mm"]
    required_dim_keys = ["length", "width", "height"]
    for key in required_dim_keys:
        if key not in dims:
            raise ValueError(f"Missing dimension field in request JSON: '{key}'")

        value = dims[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Dimension '{key}' must be a number, got {value!r} ({type(value).__name__})"
            )
        if value <= 0:
            raise ValueError(
                f"Dimension '{key}' must be a positive number, got {value}"
            )

    # Resolve image path
    image_path = request_data["image_path"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(project_root, image_path)

    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    return image_path


# ---------------------------------------------------
# 3. CREATE REQUEST OUTPUT FOLDER
# ---------------------------------------------------
def create_request_folder(project_root, request_id):
    request_folder = os.path.join(project_root, "outputs", "requests", request_id)
    if os.path.exists(request_folder) and os.listdir(request_folder):
        print(
            f"\n⚠️  WARNING: request_id '{request_id}' already exists and will be "
            f"OVERWRITTEN at {request_folder}. Use a unique request_id to avoid this."
        )
    os.makedirs(request_folder, exist_ok=True)
    return request_folder


# ---------------------------------------------------
# 4. COPY INPUTS INTO REQUEST FOLDER
# ---------------------------------------------------
def copy_request_inputs(request_json_path, image_abs_path, request_folder):
    """Saves request JSON into the request folder. The original image is
    already written by image_processor.process_image() as original_image<ext>,
    so we do NOT duplicate it here."""
    request_json_copy = os.path.join(request_folder, "input_request.json")
    shutil.copy2(request_json_path, request_json_copy)
    return request_json_copy


# ---------------------------------------------------
# 5. COPY GENERATED OUTPUTS INTO REQUEST FOLDER
# ---------------------------------------------------
def copy_generated_outputs(quote_result, request_folder):
    """Copies cost engine outputs into the request folder using the
    Phase 2 spec filenames: bom.csv, cost.csv, quote_summary.json."""
    output_files = quote_result["output_files"]

    copied_paths = {}

    # BOM  →  bom.csv
    if output_files.get("bom_csv") and os.path.exists(output_files["bom_csv"]):
        dst = os.path.join(request_folder, "bom.csv")
        shutil.copy2(output_files["bom_csv"], dst)
        copied_paths["bom"] = dst

    # Cost CSV  →  cost.csv
    if output_files.get("cost_csv") and os.path.exists(output_files["cost_csv"]):
        dst = os.path.join(request_folder, "cost.csv")
        shutil.copy2(output_files["cost_csv"], dst)
        copied_paths["cost"] = dst

    # Summary JSON  →  quote_summary.json
    if output_files.get("summary_json") and os.path.exists(output_files["summary_json"]):
        dst = os.path.join(request_folder, "quote_summary.json")
        shutil.copy2(output_files["summary_json"], dst)
        copied_paths["quote_summary"] = dst

    # Fusion scaled components  →  scaled_fusion_components.csv (bonus output)
    if output_files.get("fusion_component_report") and output_files["fusion_component_report"]:
        if os.path.exists(output_files["fusion_component_report"]):
            dst = os.path.join(request_folder, "scaled_fusion_components.csv")
            shutil.copy2(output_files["fusion_component_report"], dst)
            copied_paths["scaled_fusion_components"] = dst

    return copied_paths


# ---------------------------------------------------
# 6. SAVE REQUEST SUMMARY
# ---------------------------------------------------
def save_request_summary(request_folder, request_data, image_abs_path, quote_result, copied_paths):
    summary_path = os.path.join(request_folder, "request_summary.json")

    request_summary = {
        "request_info": {
            "request_id": request_data["request_id"],
            "customer_name": request_data["customer_name"],
            "sofa_type": request_data["sofa_type"],
            "image_path_original": image_abs_path,
            "dimensions_mm": request_data["dimensions_mm"]
        },
        "quotation_result": quote_result,
        "bundled_files": copied_paths
    }

    with open(summary_path, "w") as f:
        json.dump(request_summary, f, indent=4)

    return summary_path


# ---------------------------------------------------
# 7. MAIN
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate sofa quotation from request JSON")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to request JSON file"
    )
    args = parser.parse_args()

    # project root = parent of src/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    request_json_path = args.input

    if not os.path.isabs(request_json_path):
        # Try relative to current working directory first (most intuitive),
        # then fall back to project root (so old-style calls still work).
        cwd_candidate = os.path.abspath(request_json_path)
        root_candidate = os.path.abspath(os.path.join(project_root, request_json_path))

        if os.path.exists(cwd_candidate):
            request_json_path = cwd_candidate
        elif os.path.exists(root_candidate):
            request_json_path = root_candidate
        else:
            request_json_path = cwd_candidate  # fall through to the error below
    else:
        request_json_path = os.path.abspath(request_json_path)

    if not os.path.exists(request_json_path):
        raise FileNotFoundError(f"Request JSON not found: {request_json_path}")

    # Load request
    request_data = load_request_json(request_json_path)

    # Validate request + image
    image_abs_path = validate_request(request_data, project_root)

    request_id = request_data["request_id"]
    customer_name = request_data["customer_name"]
    sofa_type = request_data["sofa_type"]
    dims = request_data["dimensions_mm"]

    length_mm = dims["length"]
    width_mm = dims["width"]
    height_mm = dims["height"]

    print("\n===== REQUEST RECEIVED =====")
    print(f"Request ID   : {request_id}")
    print(f"Customer     : {customer_name}")
    print(f"Sofa Type    : {sofa_type}")
    print(f"Image Path   : {image_abs_path}")
    print(f"Dimensions   : L={length_mm} mm, W={width_mm} mm, H={height_mm} mm")

    # Create request folder
    request_folder = create_request_folder(project_root, request_id)

    # Copy request JSON into request folder (image is handled by image_processor in intake.py)
    copy_request_inputs(request_json_path, image_abs_path, request_folder)

    # Generate quotation
    engine = SofaCostEngine()
    quote_result = engine.generate_quote(
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        output_prefix=request_id
    )

    # Copy generated outputs into request folder
    copied_paths = copy_generated_outputs(quote_result, request_folder)

    # Save bundled request summary
    request_summary_path = save_request_summary(
        request_folder=request_folder,
        request_data=request_data,
        image_abs_path=image_abs_path,
        quote_result=quote_result,
        copied_paths=copied_paths
    )

    print("\n===== QUOTATION GENERATED =====")
    print(json.dumps(quote_result, indent=4))
    print(f"\nRequest summary saved to: {request_summary_path}")
    print(f"Bundled request folder created at: {request_folder}")


if __name__ == "__main__":
    main()