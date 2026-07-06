import os
import json
import argparse
from cost_engine import SofaCostEngine


def load_request_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def validate_request(request_data, project_root):
    required_top_keys = ["request_id", "customer_name", "sofa_type", "image_path", "dimensions_mm"]
    for key in required_top_keys:
        if key not in request_data:
            raise ValueError(f"Missing required key in request JSON: {key}")

    dims = request_data["dimensions_mm"]
    for key in ["length", "width", "height"]:
        if key not in dims:
            raise ValueError(f"Missing dimension '{key}' inside dimensions_mm")

    sofa_type = request_data["sofa_type"]
    if sofa_type != "3_seater":
        raise ValueError(
            f"Currently only '3_seater' is supported. Got sofa_type='{sofa_type}'"
        )

    image_rel_path = request_data["image_path"]
    image_abs_path = os.path.join(project_root, image_rel_path)

    if not os.path.exists(image_abs_path):
        raise FileNotFoundError(f"Image file not found: {image_abs_path}")

    return image_abs_path


def save_request_summary(project_root, request_data, quote_result):
    requests_dir = os.path.join(project_root, "outputs", "requests")
    os.makedirs(requests_dir, exist_ok=True)

    request_id = request_data["request_id"]
    summary_path = os.path.join(requests_dir, f"{request_id}_request_summary.json")

    combined = {
        "request": request_data,
        "quote_result": quote_result
    }

    with open(summary_path, "w") as f:
        json.dump(combined, f, indent=4)

    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Run sofa quotation from image + dimensions JSON request")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to request JSON file (example: ../data/sample_inputs/sample_request.json)"
    )
    args = parser.parse_args()

    # project_root = sofa_project/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    request_json_path = args.input
    if not os.path.isabs(request_json_path):
        request_json_path = os.path.join(os.path.dirname(__file__), request_json_path)

    request_json_path = os.path.abspath(request_json_path)

    if not os.path.exists(request_json_path):
        raise FileNotFoundError(f"Request JSON not found: {request_json_path}")

    request_data = load_request_json(request_json_path)

    # Validate request and image existence
    image_abs_path = validate_request(request_data, project_root)

    request_id = request_data["request_id"]
    sofa_type = request_data["sofa_type"]
    dims = request_data["dimensions_mm"]

    length_mm = dims["length"]
    width_mm = dims["width"]
    height_mm = dims["height"]

    print("\n===== REQUEST RECEIVED =====")
    print(f"Request ID   : {request_id}")
    print(f"Customer     : {request_data['customer_name']}")
    print(f"Sofa Type    : {sofa_type}")
    print(f"Image Path   : {image_abs_path}")
    print(f"Dimensions   : L={length_mm} mm, W={width_mm} mm, H={height_mm} mm")

    # Run quotation
    engine = SofaCostEngine()
    quote_result = engine.generate_quote(
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        output_prefix=request_id
    )

    request_summary_path = save_request_summary(project_root, request_data, quote_result)

    print("\n===== QUOTATION GENERATED =====")
    print(json.dumps(quote_result, indent=4))
    print(f"\nRequest summary saved to: {request_summary_path}")


if __name__ == "__main__":
    main()