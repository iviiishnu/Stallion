from ultralytics import YOLO

model = YOLO(
    "runs/segment/runs/sofa_segmentation/weights/last.pt"
)

model.train(resume=True)

print("Training resumed/completed.")