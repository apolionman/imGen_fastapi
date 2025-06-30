from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from PIL import Image
import tempfile, os, httpx, asyncio, subprocess, whisper, requests, sys, torch, io
from diffusers import FluxPipeline
from app.scripts.flux_run import *
stt = whisper.load_model("large")

sys.path.append('src/blip')
sys.path.append('clip-interrogator')

from clip_interrogator import Config, Interrogator

# --- CONFIGURATION FIX ---

config = Config()
config.device = "cuda"
config.clip_model_name = "ViT-H-14/laion2b_s32b_b79k"
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

ALLOWED_TYPES = {
    "audio/mpeg", "audio/webm", "video/mp4",
    "audio/mp4", "video/webm", "audio/x-m4a", 
    "audio/m4a",  "audio/ogg"
}

MIME_EXTENSION_MAP = {
    "audio/mpeg": ".mp3",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "video/mp4": ".mp4",
    "audio/mp4": ".mp4",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/ogg": ".ogg"
}

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
):
    """
    Transcribe uploaded audio/video using Whisper locally.
    Supports: audio/mpeg, audio/webm, video/mp4, audio/mp4, video/webm, audio/x-m4a
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Must be one of {ALLOWED_TYPES}",
        )

    suffix = MIME_EXTENSION_MAP.get(file.content_type, ".tmp")
    input_tmp = None
    wav_path = None

    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_tmp:
            input_tmp.write(await file.read())
            input_path = input_tmp.name

        # Convert to mono WAV at 24kHz for Whisper
        wav_path = input_path.replace(suffix, ".wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "24000", "-ac", "1", wav_path
        ], check=True)

        # Transcribe using whisper
        result = stt.transcribe(wav_path, fp16=False)
        text = result["text"].strip()
        return JSONResponse(content={"transcription": text})

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in [input_tmp.name if input_tmp else None, wav_path]:
            if path and os.path.exists(path):
                os.remove(path)

@router.post("/generate-image")
async def generate_image(prompt: str):
    image_path = await generate_image_task(prompt=prompt)
    return FileResponse(image_path, media_type="image/png", filename=os.path.basename(image_path))