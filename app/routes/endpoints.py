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
config.chunk_size = 2048
config.flavor_intermediate_count = 512
config.blip_num_beams = 16
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.clip_model_jit = False

# # Critical: Set default dtype before creating Interrogator
# torch.set_default_dtype(torch.float32)

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
    try:
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((512, 512)).copy()  # Force safe size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Run in thread-safe executor
    import asyncio
    from functools import partial
    loop = asyncio.get_event_loop()

    if mode == 'best':
        result = await loop.run_in_executor(None, partial(ci.interrogate, image, max_flavors=best_max_flavors))
    elif mode == 'classic':
        result = await loop.run_in_executor(None, partial(ci.interrogate_classic, image))
    elif mode == 'fast':
        result = await loop.run_in_executor(None, partial(ci.interrogate_fast, image))
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    return JSONResponse(content={"result": result})

