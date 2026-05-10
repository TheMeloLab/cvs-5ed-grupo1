import os
from ultralytics import YOLO

# Sobe 3 níveis: train/ -> src/ -> raiz do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


model = YOLO("yolov8s.pt")

results = model.train(
    data=os.path.join(BASE_DIR, "dataset", "data.yaml"),
    epochs=100,
    imgsz=640,
    name="alpr",
    patience=15,
    batch=32,
    lr0=0.01,
    augment=True,
    mosaic=1.0,
    degrees=10.0,
    hsv_v=0.25,
    fliplr=0.0,
    device=device,
)

print("Treino concluído.")
print("Melhor modelo em: runs/detect/alpr/weights/best.pt")