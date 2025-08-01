from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from typing import Optional
import os, base64, io, tempfile, json
from redis import Redis
from rq import Queue
from rq.job import Job
from app.scripts.flux_tx2im import generate_image_task
from app.scripts.flux_im2im import generate_im2im_task
import cairosvg, requests, uuid, tempfile, shutil

router = APIRouter()

# Redis setup
redis_conn = Redis(host="redis", port=6379)
queue = Queue("flux_image_gen", connection=redis_conn, default_timeout=3600)

@router.get("/health")
async def health():
    return {"status": "ok"}

class FluxRequest(BaseModel):
    prompt: str
    task_id: str
    user_uuid: str

@router.post("/generate-flux")
async def enqueue_flux_task(req: FluxRequest):
    # Step 1: Generate a unique task ID
    # task_id = str(uuid.uuid4())
    # Step 2: Enqueue the job with the custom task ID and additional arguments
    job = queue.enqueue(
        generate_image_task,
        req.prompt,
        req.user_uuid,
        req.task_id,
        job_id=req.task_id  # Set job ID explicitly
    )
    return {"task_id": job.id, "status": "queued"}

class FluxImageRequest(BaseModel):
    prompt: str
    task_id: str
    user_uuid: str
    image_url: str

@router.post("/generate-flux-im2im")
async def enqueue_flux_im2im(req: FluxImageRequest):
    try:
        # Download image
        response = requests.get(req.image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from URL.")
        contents = response.content

        # Save image to ./input folder with unique filename
        input_folder = '/app/input_images'
        os.makedirs(input_folder, exist_ok=True)

        input_id = str(uuid.uuid4())[:8]
        ext = ".png"
        input_path = os.path.join(input_folder, f"{input_id}{ext}")

        # Handle SVG vs other images
        if req.image_url.lower().endswith(".svg"):
            cairosvg.svg2png(bytestring=contents, write_to=input_path)
        else:
            try:
                image = Image.open(io.BytesIO(contents)).convert("RGB")
            except UnidentifiedImageError:
                raise HTTPException(status_code=400, detail="Unsupported image format.")
            image.save(input_path, format="PNG")

        # Enqueue job with full image path
        job = queue.enqueue(
            generate_im2im_task,
            req.prompt,
            input_path,
            req.user_uuid,
            req.task_id
        )

        return {"task_id": job.id, "status": "queued"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
