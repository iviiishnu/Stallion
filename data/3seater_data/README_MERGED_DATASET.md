# Stallion Sofa Master YOLOv8 Segmentation Dataset

## Final dataset
- Train images: 116
- Validation images: 42
- Total unique images: 158
- Classes: 7
- Total annotated objects: 1337

## Classes
0. back_cushion
1. base
2. left_arm
3. legs
4. right_arm
5. seat_cushion
6. stallion-sofa

## What was cleaned
1. Three supplied YOLOv8 datasets were merged.
2. Class IDs were normalized to one master class vocabulary.
3. `stallion-sofa1`, `stallion-sofa2`, and `stallion-sofa` were unified as `stallion-sofa`.
4. Seven repeated source images shared by the first two datasets were deduplicated.
   The v3 copy was retained to preserve its original split and avoid leakage.
5. Seven bounding-box annotations in one source image were converted into rectangular
   segmentation polygons so all labels are valid YOLO segmentation labels.
6. Images and labels were renamed with unique deterministic IDs.
7. A manifest, class mapping, duplicate log, conversion log, and QA report are included.

## Training
Use `data.yaml` with a YOLOv8 segmentation model, for example a `*-seg.pt` checkpoint.

There is intentionally no fabricated test set. If you need a held-out test set later,
create it from the training set only after deciding the evaluation protocol.

## Important annotation note
The seven converted annotations are rectangular polygons derived from bounding boxes.
They are technically valid segmentation labels, but they are not equivalent to manually
traced segmentation masks. The conversion log identifies every one of them.

## Provenance
See `SOURCE_ATTRIBUTION.txt` and `manifest.csv` for source-level traceability.
