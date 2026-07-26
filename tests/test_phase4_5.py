"""
Stallion Phase 4 & 5 Test Suite
================================
Runs automated tests for:
  - Phase 4: Dimension validation limits (all sofa types)
  - Phase 4: BOM scaling accuracy per sofa type
  - Phase 5: Generates a test JSON output for Fusion 360 StallionLink

Run from the Stallion root directory:
    python tests/test_phase4_5.py
"""

import sys
import os
import json
import io

# Force UTF-8 output on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cost_engine import SofaCostEngine

PASS = "  [PASS]"
FAIL = "  [FAIL]"
HEAD = "\n=================================="

engine = SofaCostEngine()
engine.load_data()

results = {"passed": 0, "failed": 0}

def check(label, fn):
    try:
        fn()
        print(f"{PASS}  {label}")
        results["passed"] += 1
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"         → {e}")
        results["failed"] += 1

def expect_raises(label, fn, error_type=ValueError):
    try:
        fn()
        print(f"{FAIL}  {label} (should have raised an error but did not)")
        results["failed"] += 1
    except error_type as e:
        print(f"{PASS}  {label}")
        print(f"         → Correctly raised: {e}")
        results["passed"] += 1
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"         → Unexpected error: {e}")
        results["failed"] += 1


# ----------------------------------------------
print(HEAD)
print("  PHASE 4: DIMENSION LIMITS VALIDATION")
print(HEAD)

# Valid in-bounds tests
check("1-Seater valid dims (850x800x800)",
    lambda: engine.generate_quote(850, 800, 800, sofa_type='1-seater', output_prefix='test'))

check("2-Seater valid dims (1500x900x850)",
    lambda: engine.generate_quote(1500, 900, 850, sofa_type='2-seater', output_prefix='test'))

check("3-Seater valid dims (2100x900x850)",
    lambda: engine.generate_quote(2100, 900, 850, sofa_type='3-seater', output_prefix='test'))

check("4-Seater valid dims (2700x950x850)",
    lambda: engine.generate_quote(2700, 950, 850, sofa_type='4-seater', output_prefix='test'))

check("L-Shape valid dims (2800x1000x850)",
    lambda: engine.generate_quote(2800, 1000, 850, sofa_type='l-shape', output_prefix='test'))

print()

# Out-of-bounds rejection tests
expect_raises("1-Seater rejects Length=1200 (max=1000)",
    lambda: engine.generate_quote(1200, 800, 800, sofa_type='1-seater', output_prefix='test'))

expect_raises("2-Seater rejects Length=500 (min=1300)",
    lambda: engine.generate_quote(500, 900, 850, sofa_type='2-seater', output_prefix='test'))

expect_raises("3-Seater rejects Width=600 (min=850)",
    lambda: engine.generate_quote(2100, 600, 850, sofa_type='3-seater', output_prefix='test'))

expect_raises("4-Seater rejects Height=600 (min=750)",
    lambda: engine.generate_quote(2700, 950, 600, sofa_type='4-seater', output_prefix='test'))

expect_raises("L-Shape rejects Length=4000 (max=3200)",
    lambda: engine.generate_quote(4000, 1000, 850, sofa_type='l-shape', output_prefix='test'))


# ----------------------------------------------
print(HEAD)
print("  PHASE 4: BOM SCALING PER SOFA TYPE")
print(HEAD)

def test_bom_scaling(sofa_type, length, width, height):
    scales, bom_df = engine.generate_scaled_bom(length, width, height, sofa_type=sofa_type)
    cost_df, summary = engine.compute_cost(bom_df)
    assert not bom_df.empty, "BOM dataframe is empty"
    assert summary["final_quotation_price"] > 0, "Final price is 0 or negative"
    price = summary["final_quotation_price"]
    print(f"           Final Price: ₹{price:,.2f}")

for (sofa_type, l, w, h) in [
    ("1-seater", 850, 800, 800),
    ("2-seater", 1500, 900, 850),
    ("3-seater", 2100, 900, 850),
    ("4-seater", 2700, 950, 850),
    ("l-shape",  2800, 1000, 850),
]:
    check(f"{sofa_type} BOM scales and produces a valid price",
        lambda s=sofa_type, l=l, w=w, h=h: test_bom_scaling(s, l, w, h))


# ----------------------------------------------
print(HEAD)
print("  PHASE 5: GENERATE FUSION 360 TEST JSON")
print(HEAD)

# Generate a test quote and produce the output JSON for StallionLink testing
os.makedirs('outputs/requests', exist_ok=True)
test_request = {
    "request_id": "fusion_test_001",
    "customer_name": "Test Customer",
    "sofa_type": "3_seater",
    "image_path": "test_image.jpg",
    "dimensions_mm": {
        "length": 2100,
        "width": 900,
        "height": 850
    }
}
json_path = os.path.abspath("outputs/requests/fusion_test_001_input_request.json")
with open(json_path, 'w') as f:
    json.dump(test_request, f, indent=4)

check("Phase 5: Test JSON written for StallionLink",
    lambda: None if os.path.exists(json_path) else (_ for _ in ()).throw(FileNotFoundError("JSON not written")))

print(f"\n         📄 JSON path: {json_path}")
print(f"         → Load this file into the StallionLink script in Fusion 360 to test Phase 5.")


# ----------------------------------------------
print(HEAD)
total = results["passed"] + results["failed"]
print(f"  RESULTS: {results['passed']}/{total} tests passed")
if results["failed"] == 0:
    print("  All tests passed!")
else:
    print(f"  {results['failed']} test(s) failed")
print()
