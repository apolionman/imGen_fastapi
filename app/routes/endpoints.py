from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Dict, Optional
import tempfile, os, httpx, asyncio, subprocess, sys, io
from PIL import Image
from uuid import uuid4
from datetime import datetime, timedelta

sys.path.append('src/blip')
sys.path.append('clip-interrogator')

from clip_interrogator import Config, Interrogator

# Initialize once at startup
config = Config()
config.blip_offload = True
config.chunk_size = 2048
config.flavor_intermediate_count = 512
config.blip_num_beams = 64

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
    # Read image bytes
    try:
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Run inference based on mode
    if mode == 'best':
        result = ci.interrogate(image, max_flavors=best_max_flavors)
    elif mode == 'classic':
        result = ci.interrogate_classic(image)
    elif mode == 'fast':
        result = ci.interrogate_fast(image)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode, choose from 'best', 'classic', 'fast'")

    return JSONResponse(content={"result": result})