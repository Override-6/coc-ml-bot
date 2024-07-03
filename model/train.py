from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

IMAGE_SIZE = 2000

model.train(
    data="../../coc-base-generator/out/data.yaml",
    epochs=100000,
    batch=2,
    imgsz=IMAGE_SIZE,
    augment=True,
)

