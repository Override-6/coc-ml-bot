from ultralytics import YOLO

model = YOLO("runs/detect/train41/weights/best.pt")
# model = YOLO("yolov8n.pt")

IMAGE_SIZE = 640

model.train(
    data="../../coc-base-generator/out/data.yaml",
    epochs=100000,
    batch=2,
    imgsz=IMAGE_SIZE,
    augment=False,
)

