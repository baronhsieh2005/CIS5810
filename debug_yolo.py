import cv2
from ultralytics import YOLO

# Load model
yolo_model = YOLO("yolo11l-pose.pt")

# Load image (your bench press image)
img_np = cv2.imread("images/not flared/nf1.png")

# Run YOLO pose with saving turned on
results = yolo_model.predict(img_np, save=True, save_txt=True)

print("Annotated image saved in: runs/pose/predict/")
