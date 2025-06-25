from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import io
from typing import Optional
from PIL import Image
import torch
import sys
import requests

sys.path.append('src/blip')
sys.path.append('clip-interrogator')

from clip_interrogator import Config, Interrogator

# --- CONFIGURATION FIX ---

config = Config()
config.device = "cuda"
config.clip_model_name = "ViT-L-14/openai"
config.precision = 'full'  
ci = Interrogator(config)

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/interrogate")
async def interrogate_image(
    image_file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    # Check if either file or URL is provided
    if not image_file and not image_url:
        raise HTTPException(status_code=400, detail="Provide either an image file or an image URL.")

    try:
        if image_file:
            image_bytes = await image_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif image_url:
            response = requests.get(image_url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch image from URL.")
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image input: {e}")

    try:
        result = ci.interrogate(image)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return JSONResponse(content={"result": result})