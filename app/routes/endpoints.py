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

@router.get("/generate-flux/status/{task_id}")
async def flux_task_status(task_id: str, return_base64: Optional[bool] = True):
    try:
        job = Job.fetch(task_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Task not found")

    if job.is_finished:
        result = job.result
        image_path = result.get("image_path")
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=500, detail="Image not found")

        if return_base64:
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            return JSONResponse(content={
                "status": "success",
                "image_base64": f"data:image/png;base64,{encoded_string}",
                "seed": result.get("seed")
            })

        return FileResponse(
            path=image_path,
            media_type="image/png",
            filename=os.path.basename(image_path)
        )
    elif job.is_failed:
        return {"status": "failed"}
    else:
        return {"status": job.get_status()}

class FluxImageRequest(BaseModel):
    prompt: str
    task_id: str
    user_uuid: str
    image_url: str

@router.post("/generate-flux-im2im")
async def enqueue_flux_im2im(req: FluxImageRequest):
    tmp_file = None

    try:
        # Download image
        response = requests.get(req.image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from URL.")
        contents = response.content

        # Create temp file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp_file.name
        tmp_file.close()  # Close file so it can be written

        # Handle SVG vs image
        if req.image_url.lower().endswith(".svg"):
            cairosvg.svg2png(bytestring=contents, write_to=tmp_path)
        else:
            try:
                image = Image.open(io.BytesIO(contents)).convert("RGB")
            except UnidentifiedImageError:
                raise HTTPException(status_code=400, detail="Unsupported image format.")
            image.save(tmp_path, format="PNG")

        # Enqueue the job
        job = queue.enqueue(
            generate_im2im_task,
            req.prompt,
            tmp_path,
            req.user_uuid,
            req.task_id,
            job_id=req.task_id
        )

        return {"task_id": job.id, "status": "queued"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate-flux-im2im/status/{task_id}")
async def flux_im2im_task_status(task_id: str, return_base64: Optional[bool] = True):
    try:
        job = Job.fetch(task_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Task not found")

    if job.is_finished:
        result = job.result
        image_path = result.get("image_path")
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=500, detail="Image not found")

        if return_base64:
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            return JSONResponse(content={
                "status": "success",
                "image_base64": f"data:image/png;base64,{encoded_string}",
                "seed": result.get("seed")
            })

        return FileResponse(
            path=image_path,
            media_type="image/png",
            filename=os.path.basename(image_path)
        )
    elif job.is_failed:
        return {"status": "failed"}
    else:
        return {"status": job.get_status()}
