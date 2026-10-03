from ultralytics import YOLO


# Pretrained YOLO segmentation model
model = YOLO("yolo11n-seg.pt")


# Train
results = model.train(
    data="dataset/merged_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",
    project="runs",
    name="sofa_segmentation",
    exist_ok=True
)

print("Training completed.")