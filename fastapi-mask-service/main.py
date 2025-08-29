# from typing import Optional

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}
import base64, io, os
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from PIL import Image
import cv2
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# ---------- Konfiguraatio ----------
API_KEY = os.environ.get("MASK_API_KEY", "").strip()
DEFAULT_PROMPTS = [
    "balcony", "balcony railing", "balustrade",
    "parveke", "parvekekaide"
]
# pienemmäksi skaalaus segmentoinnille (nopeus vs. tarkkuus)
SHORT_SIDE = int(os.environ.get("SHORT_SIDE", "512"))
# kynnykset
PROB_THRESHOLD = float(os.environ.get("PROB_THRESHOLD", "0.35"))
MORPH_KERNEL = int(os.environ.get("MORPH_KERNEL", "5"))
DILATE_ITERS = int(os.environ.get("DILATE_ITERS", "1"))
CLOSE_ITERS = int(os.environ.get("CLOSE_ITERS", "1"))

# ---------- FastAPI ----------
app = FastAPI(title="Parveke Mask Service", version="1.0.0")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lataa malli
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
model.eval()

# ---------- Pydantic ----------
class PredictReq(BaseModel):
    image_base64: str                # pakollinen, raw base64 (ilman data:image/png;base64, -prefiksiä)
    prompts: Optional[List[str]] = None  # valinnainen: jos haluat omat promptit tälle pyynnölle
    prob_threshold: Optional[float] = None
    morph_kernel: Optional[int] = None
    dilate_iters: Optional[int] = None
    close_iters: Optional[int] = None

class PredictResp(BaseModel):
    mask_base64: str  # 8-bit PNG, white = edit area

# ---------- Utils ----------
def decode_image(b64: str) -> Image.Image:
    try:
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")

def np_to_png_base64(arr: np.ndarray) -> str:
    # arr: uint8 (H,W) 0..255
    pil = Image.fromarray(arr)
    buff = io.BytesIO()
    pil.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def ensure_auth(auth_header: Optional[str]):
    if not API_KEY:
        return  # avainta ei vaadita
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <API_KEY>")
    token = auth_header.split(" ",1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True, "device": device}

@app.post("/predict", response_model=PredictResp)
def predict(req: PredictReq, authorization: Optional[str] = Header(None)):
    ensure_auth(authorization)

    # 1) Lue kuva
    img = decode_image(req.image_base64)
    W, H = img.size

    # 2) Skaalaa lyhyen sivun mukaan
    short = max(64, SHORT_SIDE)
    scale = short / min(W, H) if min(W, H) > short else 1.0
    target_w, target_h = int(round(W*scale)), int(round(H*scale))
    img_small = img.resize((target_w, target_h), Image.BILINEAR)

    # 3) Prompteilla segmentointi
    prompts = req.prompts if (req.prompts and len(req.prompts) > 0) else DEFAULT_PROMPTS
    with torch.no_grad():
        inputs = processor(
            text=prompts,
            images=[img_small] * len(prompts),
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        logits = outputs.logits.sigmoid().cpu().numpy()  # (N, H, W) ∈ [0,1]
        prob = np.max(logits, axis=0)                    # yhdistä maxilla prompttien yli

    # 4) Takaisin alkuperäiseen kokoon
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_CUBIC)

    # 5) Kynnys + morfologia (siistiminen)
    th = float(req.prob_threshold or PROB_THRESHOLD)
    mask_bin = (prob >= th).astype(np.uint8) * 255

    k = int(req.morph_kernel or MORPH_KERNEL)
    kernel = np.ones((max(1,k), max(1,k)), np.uint8)

    ci = int(req.close_iters or CLOSE_ITERS)
    di = int(req.dilate_iters or DILATE_ITERS)
    if ci > 0:
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=ci)
    if di > 0:
        mask_bin = cv2.dilate(mask_bin, kernel, iterations=di)

    # 6) Palauta WHITE = edit area (lisäosa muuntaa tämän OpenAI-semanttiikaksi)
    return PredictResp(mask_base64=np_to_png_base64(mask_bin))