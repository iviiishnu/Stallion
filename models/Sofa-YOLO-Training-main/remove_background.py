import os
from pathlib import Path

from PIL import Image
from rembg import remove
from ultralytics import YOLO


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = r"models\best.pt"
# ============================================================
# ASK USER FOR IMAGE PATH
# ============================================================

image_path = input("\nEnter the path of the sofa image: ").strip().strip('"')

if not os.path.isfile(image_path):
    print("\nERROR: Image file does not exist.")
    print("Please check the path and try again.")
    exit()


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.isfile(MODEL_PATH):
    print("\nERROR: Trained YOLO model not found:")
    print(MODEL_PATH)
    exit()


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

output_folder = Path("runs") / "complete_pipeline"
output_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1 - REMOVE BACKGROUND
# ============================================================

print("\n========================================")
print("STEP 1: Removing background...")
print("========================================")

try:
    original_image = Image.open(image_path).convert("RGBA")

    # Remove background
    removed = remove(original_image)

    # Create white background
    white_background = Image.new(
        "RGBA",
        removed.size,
        (255, 255, 255, 255)
    )

    # Put sofa over white background
    final_image = Image.alpha_composite(
        white_background,
        removed
    )

    # Convert to RGB for YOLO
    final_image = final_image.convert("RGB")

    background_removed_path = (
        output_folder / "background_removed.jpg"
    )

    final_image.save(background_removed_path, quality=95)

    print("\nBackground removed successfully.")
    print("Saved to:")
    print(background_removed_path)


except Exception as e:
    print("\nERROR during background removal:")
    print(e)
    exit()


# ============================================================
# STEP 2 - COMPONENT DETECTION
# ============================================================

print("\n========================================")
print("STEP 2: Detecting sofa components...")
print("========================================")

try:
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=str(background_removed_path),
        conf=0.25,
        save=True,
        project=str(output_folder),
        name="component_detection",
        exist_ok=True
    )

except Exception as e:
    print("\nERROR during component detection:")
    print(e)
    exit()


# ============================================================
# DISPLAY DETECTED COMPONENTS
# ============================================================

print("\n========================================")
print("DETECTED COMPONENTS")
print("========================================")

for result in results:

    if result.boxes is None:
        print("No components detected.")
        continue

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(
            f"{class_name}: "
            f"{confidence:.2f}"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

final_output_folder = (
    output_folder / "component_detection"
)

print("\n========================================")
print("PIPELINE COMPLETED")
print("========================================")

print("\nBackground-removed image:")
print(background_removed_path)

print("\nFinal component detection:")
print(final_output_folder)

print("\nThe final image contains:")
print("- Background removed")
print("- Sofa isolated")
print("- Component detection")
print("- Component labels")
print("- Confidence scores")
print("- Segmentation masks")