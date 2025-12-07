import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel


from model_setup import (
    get_yolo_keypoints_from_pil,
    get_dino_patch_features,
    kpts_to_patch_indices,
    compute_frame_embedding_from_pil,
    predict_phase_from_pil,
)

from typing import Tuple
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from form_analysis import (
    analyze_elbow_line,
    classify_flare

)

class BenchPhaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, hidden_dim2=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

DINO_LOCAL_DIR = "./dinov3_local"

processor = AutoImageProcessor.from_pretrained(DINO_LOCAL_DIR)
dino_model = AutoModel.from_pretrained(DINO_LOCAL_DIR).to(device)
dino_model.eval()
patch_size = dino_model.config.patch_size

yolo_model = YOLO("yolo11l-pose.pt")  

CKPT_PATH = "./bench_phase_mlp.pt" 
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
input_dim = ckpt["input_dim"]
classifier_model = BenchPhaseMLP(input_dim=input_dim).to(device)
classifier_model.load_state_dict(ckpt["state_dict"])
classifier_model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    label, probs = predict_phase_from_pil(
        img_pil,
        yolo_model=yolo_model,
        dino_model=dino_model,
        processor=processor,
        patch_size=patch_size,
        classifier_model=classifier_model,
        device=device,
    )

    if label is None:
        raise HTTPException(status_code=422, detail="No person detected in image")

    return {
        "phase": label,
        "probabilities": {
            "lowering": float(probs[0]),
            "pushing": float(probs[1]),
        },
    }

"""
@app.post("/analyze_form")
async def analyze_bench_form(file: UploadFile = File(...)):
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img_pil)

    # Run YOLO Pose
    results = yolo_model.predict(img_np, save=False, verbose=False)
    if len(results) == 0 or len(results[0].keypoints.xy) == 0:
        raise HTTPException(status_code=422, detail="No person detected")

    kpts = results[0].keypoints.xy[0].cpu().numpy()

    # Run form analysis
    analysis = analyze_form(kpts)

    return analysis

"""

@app.post("/elbow_line_angle")
async def elbow_line_angle_api(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)
    results = yolo_model.predict(img_np, verbose=False)
    kpts = results[0].keypoints.xy[0].cpu().numpy()
    return analyze_elbow_line(kpts)

@app.post("/elbow_flare")
async def elbow_flare_api(file: UploadFile = File(...), threshold: float = 40.0):
    """
    Classifies elbow flare from a BENCH-PRESS LOWERED POSITION image.
    Assumes img contains a clear bottom-position rep.
    """

    # Read img
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_np = np.array(img)

    # YOLO Pose Detection
    results = yolo_model.predict(img_np, verbose=False)
    if len(results) == 0 or len(results[0].keypoints.xy) == 0:
        raise HTTPException(status_code=422, detail="No person detected")

    # Get keypoints for first detected person
    kpts = results[0].keypoints.xy[0].cpu().numpy()  # shape (17,2)

    # Flare classification from helper file
    analysis = classify_flare(kpts, side_threshold=threshold)

    return analysis
