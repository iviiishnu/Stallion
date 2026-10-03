from pathlib import Path

LABEL_DIR = Path("dataset/merged_dataset/train/labels")

fixed_files = 0
removed_lines = 0

for label_file in LABEL_DIR.glob("*.txt"):
    lines = label_file.read_text().splitlines()

    valid_lines = []

    for line in lines:
        values = line.strip().split()

        # YOLO segmentation:
        # class + at least 3 coordinate pairs
        if len(values) >= 7 and (len(values) - 1) % 2 == 0:
            valid_lines.append(line)
        else:
            removed_lines += 1

    if len(valid_lines) != len(lines):
        label_file.write_text(
            "\n".join(valid_lines) + ("\n" if valid_lines else "")
        )
        fixed_files += 1

print("Label fixing completed.")
print(f"Files modified: {fixed_files}")
print(f"Invalid lines removed: {removed_lines}")