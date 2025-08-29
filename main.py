# from typing import Optional

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}
import base64, io
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import cv2

import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ladataan malli
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
model.eval()

PROMPTS = [
    "balcony", "balcony railing", "balustrade",
    "parveke", "parvekekaide"
]

class Req(BaseModel):
    image_base64: str  # data ilman prefixiä, pelkkä base64

def pil_to_np(img):
    return np.array(img)

def np_to_png_bytes(arr):
    pil = Image.fromarray(arr)
    buff = io.BytesIO()
    pil.save(buff, format="PNG")
    return buff.getvalue()

@app.post("/predict")
def predict(req: Req):
    # Decode image
    img_bytes = base64.b64decode(req.image_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = img.size

    # Mallin resoluutio
    short = 512
    scale = short / min(W, H)
    target_w, target_h = int(round(W*scale)), int(round(H*scale))
    img_small = img.resize((target_w, target_h), Image.BILINEAR)

    # Kokoa usean promptin maksimi
    with torch.no_grad():
        inputs = processor(
            text=PROMPTS,
            images=[img_small] * len(PROMPTS),
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        # outputs.logits: (N, H, W) pienessä koossa
        logits = outputs.logits.sigmoid().cpu().numpy()  # [0,1]
        # Maksimi yli promptien
        prob = np.max(logits, axis=0)

    # Uudelleenskaalaus takaisin alkuperäiseen kokoon
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_CUBIC)

    # Kynnys + siistiminen
    th = 0.35
    mask_bin = (prob >= th).astype(np.uint8) * 255

    # morfologia: laajennus ja aukotus
    kernel = np.ones((5,5), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)

    # Palauta WHITE = edit area (lisäosan normalize_mask_for_openai käyttää white_means_edit=true)
    # 8-bit PNG, valkoinen = muokkaa, musta = pidä
    png_bytes = np_to_png_bytes(mask_bin)
    mask_b64 = base64.b64encode(png_bytes).decode("utf-8")

    return {"mask_base64": mask_b64}