from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    name="alpr",
    patience=15,
    batch=32,
    lr0=0.01,
    augment=True,
    mosaic=1.0,
    degrees=10.0,
    hsv_v=0.4,
    fliplr=0.0,
    device="mps",
)

print("Treino concluído.")
print("Melhor modelo em: models/best.pt")