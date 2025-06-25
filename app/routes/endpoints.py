from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import io
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
    image_file: UploadFile = File(...),
    mode: str = Form('best'),
    best_max_flavors: int = Form(5),
):
    try:
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Run inference based on mode
    # The ci.interrogate methods will handle device placement internally
    try:
        result = ci.interrogate(image)
        # if mode == 'best':
        #     result = ci.interrogate(image, max_flavors=best_max_flavors)
        # elif mode == 'classic':
        #     result = ci.interrogate_classic(image)
        # elif mode == 'fast':
        #     result = ci.interrogate_fast(image)
        # else:
        #     raise HTTPException(status_code=400, detail="Invalid mode, choose from 'best', 'classic', 'fast'")
    except RuntimeError as e:
        # Catch potential runtime errors from the model and return a 500
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")


    return JSONResponse(content={"result": result})