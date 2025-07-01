from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from PIL import Image
from app.scripts.flux_run import *
import tempfile, os, httpx, asyncio, subprocess, requests, sys, torch, io
# from diffusers import FluxPipeline
# from app.scripts.flux_run import *
import base64

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/generate-flux")
async def generate_flux(prompt: str, return_base64: bool = False):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, generate_image_task, prompt)

    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    image_path = result["image_path"]

    if return_base64:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        return JSONResponse(content={
            "image_base64": f"data:image/png;base64,{encoded_string}"
        })

    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=os.path.basename(image_path)
    )
