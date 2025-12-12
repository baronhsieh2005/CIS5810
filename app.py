import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from openai import OpenAI
import base64


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

SYSTEM_PROMPT = """
You are a strength coach specializing in bench press form analysis.

Your job has three parts, and you must output them in THREE SEPARATE SECTIONS:

------------------------------------------------------------
SECTION 1 — GENERAL BENCH PRESS GUIDELINES
Provide universal, high-level form cues that apply to ALL lifters.
Do NOT reference the user's specific image or data here.
Examples: scapular retraction, bar path, breathing, leg drive, etc.

------------------------------------------------------------
SECTION 2 — TAILORED FEEDBACK (BASED ONLY ON INPUT DATA)
Give personalized advice ONLY about issues that are visible in:
- the provided image solely and do not use other data that is provided, make it easier to understand for the general public (do not use
too complicated anatomy terms and treat it like giving gym feedback to a beginner trying to correct their form), and also provide more feedback 
as well as words of encouragement
------------------------------------------------------------
SECTION 3 — EXPLANATIONS (CITE THE INPUT VALUES)
For EACH tailored correction you gave in Section 2:
- Explain exactly which metric(s) or features in the image caused you to give that advice.
- Include the numeric values (angles, thresholds, booleans, etc).
- NEVER invent data.
- Only refer to the inputs provided.

Your output must follow this structure exactly:

SECTION 1: General Guidelines
...

SECTION 2: Tailored Feedback
...

SECTION 3: Explanations
...
"""


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

@app.post("/elbow_line_angle")
async def elbow_line_angle_api(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)
    results = yolo_model.predict(img_np, verbose=False)
    kpts = results[0].keypoints.xy[0].cpu().numpy()
    return analyze_elbow_line(kpts)


@app.post("/elbow_flare")
async def elbow_flare_api(
    file: UploadFile = File(...),
    threshold: float = 70.0,
):
    """
    Classifies elbow flare from a bench press lowered position image.
    """

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_np = np.array(img)

    results = yolo_model.predict(img_np, verbose=False)
    if len(results) == 0 or len(results[0].keypoints.xy) == 0:
        raise HTTPException(status_code=422, detail="No person detected")

    kpts = results[0].keypoints.xy[0].cpu().numpy()

    analysis = classify_flare(
        kpts,
        threshold_deg=threshold
    )

    return analysis

client = OpenAI()

@app.post("/analyze_with_llm")
async def analyze_with_llm(
    file: UploadFile = File(...),
    flare_threshold: float = 70.0,
    bar_level_threshold: float = 10.0
):
    """
    Full analysis pipeline:
      YOLO keypoints → bar level → flare → OpenAI LLM advice
    """

    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
    img_np = np.array(img)

    results = yolo_model.predict(img_np, verbose=False)
    if len(results) == 0 or results[0].keypoints is None:
        raise HTTPException(status_code=422, detail="No person detected")

    kpts = results[0].keypoints.xy[0].cpu().numpy()

    bar_analysis = analyze_elbow_line(kpts, level_threshold_deg=bar_level_threshold)

    flare_analysis = classify_flare(
        kpts,
        threshold_deg=flare_threshold
    )

    payload = {
        "keypoints": kpts.tolist()
    }

    image_b64 = base64.b64encode(contents).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
                        Follow the SYSTEM INSTRUCTIONS to produce the 3 required sections.

                        Here is the biomechanical analysis data:
                        {payload}

                        Only base tailored feedback on:
                        - the image and the keypoints

                        Do not mention or correct unrelated features.
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }

        ]
    )

    return {
        "analysis": payload,
        "llm_advice": response.choices[0].message.content
    }
