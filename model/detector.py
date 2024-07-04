from ultralytics import YOLO

# model = YOLO("../../coc-base-generator/runs/detect/train9/weights/best.pt")
model = YOLO("runs/detect/train32/weights/best.pt")

# results = model("datasets/buildings/images/train/Archer_Tower1.png")
# results = model("../../coc-base-generator/out/images/test/village_0.png")
results = model("test_images/th11_defence_97.jpg")

for result in results:
    # Display or save each result
    print(result)
    result.show()  # Display the image with detections