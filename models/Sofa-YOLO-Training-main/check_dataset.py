from pathlib import Path
from PIL import Image

DATASET = Path("dataset/merged_dataset")
NUM_CLASSES = 11


def check_split(split):
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    invalid_images = []
    invalid_labels = []

    # Check images
    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            invalid_images.append(image_path.name)

    # Check segmentation labels
    for label_path in labels:
        try:
            lines = label_path.read_text().splitlines()

            for line_number, line in enumerate(lines, 1):
                parts = line.split()

                # Segmentation needs:
                # class_id + at least 3 (x,y) points
                if len(parts) < 7:
                    invalid_labels.append(
                        f"{label_path.name}: line {line_number} - too few values"
                    )
                    continue

                class_id = int(parts[0])
                coordinates = [float(x) for x in parts[1:]]

                if not 0 <= class_id < NUM_CLASSES:
                    invalid_labels.append(
                        f"{label_path.name}: line {line_number} - invalid class {class_id}"
                    )
                    continue

                if len(coordinates) % 2 != 0:
                    invalid_labels.append(
                        f"{label_path.name}: line {line_number} - odd coordinate count"
                    )
                    continue

                if not all(0 <= value <= 1 for value in coordinates):
                    invalid_labels.append(
                        f"{label_path.name}: line {line_number} - coordinate outside 0-1"
                    )

        except Exception as error:
            invalid_labels.append(
                f"{label_path.name}: unreadable ({error})"
            )

    print(f"\n--- {split.upper()} ---")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Invalid images: {len(invalid_images)}")
    print(f"Invalid labels: {len(invalid_labels)}")

    if invalid_labels:
        print("\nInvalid labels:")
        for item in invalid_labels:
            print(item)


print("Checking YOLO segmentation dataset...")

check_split("train")
check_split("valid")

print("\nDataset check completed.")