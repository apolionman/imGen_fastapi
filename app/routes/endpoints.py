from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import io
from PIL import Image
import torch
import sys

sys.path.append('src/blip')
sys.path.append('clip-interrogator')

from clip_interrogator import Config, Interrogator

# Initialize once at startup
config = Config()
config.blip_offload = True
config.chunk_size = 512
config.flavor_intermediate_count = 64
config.blip_num_beams = 8
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.clip_model_jit = False  # Disable JIT

# Critical: Set default dtype before creating Interrogator
torch.set_default_dtype(torch.float32)

# Now create Interrogator
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

    # If possible, move the image tensor to device before inference
    # Here we assume ci.interrogate internally converts PIL image to tensor,
    # so let's patch that conversion to move to the right device

    # Monkey patch ci.interrogate to move tensors to device
    # (If you can modify clip_interrogator, better to fix it there)

    # Helper wrapper to move inputs inside ci.interrogate
    def run_on_device(func, *args, **kwargs):
        # Convert PIL image to tensor and move to device, if needed
        # But since ci.interrogate expects PIL Image, we just call func directly
        # The actual model inside should handle device, or fix in library
        return func(*args, **kwargs)

    # Run inference based on mode
    if mode == 'best':
        result = run_on_device(ci.interrogate, image, max_flavors=best_max_flavors)
    elif mode == 'classic':
        result = run_on_device(ci.interrogate_classic, image)
    elif mode == 'fast':
        result = run_on_device(ci.interrogate_fast, image)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode, choose from 'best', 'classic', 'fast'")

    return JSONResponse(content={"result": result})
