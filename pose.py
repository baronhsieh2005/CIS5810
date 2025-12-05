from ultralytics import YOLO
import os

# Load YOLO pose model
model = YOLO("yolo11x-pose.pt")

# Directory of your training images
directory_path = "./images"

# Loop through images
for entry in os.listdir(directory_path):
    img_path = os.path.join(directory_path, entry)
    print(img_path)
    # Run pose prediction
    results = model.predict(
        source=img_path,
        imgsz=640,       # try 640 or 960; 960 helps small joints a bit
        conf=0.2,        # keep more candidates, you’ll filter later
        save=True,       # save drawn images
        save_txt=True,   # save labels for training
        save_conf=True,  # store confs in txt
        show=False,
        max_det=1,       # don’t blow up if there are many people
    )
    print(f"{img_path} is annotated")

