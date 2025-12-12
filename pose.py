from ultralytics import YOLO
import os

model = YOLO("yolo11x-pose.pt")

directory_path = "./images"

for entry in os.listdir(directory_path):
    img_path = os.path.join(directory_path, entry)
    print(img_path)
    results = model.predict(
        source=img_path,
        imgsz=640,
        conf=0.2,
        save=True,
        save_txt=True,
        save_conf=True,
        show=False,
        max_det=1,
    )
    print(f"{img_path} is annotated")

