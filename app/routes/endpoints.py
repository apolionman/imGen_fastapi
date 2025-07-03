from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from typing import Optional
import asyncio, os, base64, io, tempfile
from app.scripts.flux_tx2im import generate_image_task
from app.scripts.flux_im2im import generate_im2im_task

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


# Request body model
class FluxRequest(BaseModel):
    prompt: str
    return_base64: Optional[bool] = True
    seed: Optional[int] = None

@router.post("/generate-flux")
async def generate_flux(req: FluxRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, generate_image_task, req.prompt, req.seed)

    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    image_path = result["image_path"]

    if req.return_base64:
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

@router.post("/generate-flux-im2im")
async def generate_flux_im2im(
    prompt: str = Form(...),
    input_image: UploadFile = File(...),
    return_base64: Optional[bool] = Form(True),
    seed: Optional[int] = Form(None)
):
    try:
        # Read raw image bytes
        contents = await input_image.read()

        # Detect file type
        filename = input_image.filename.lower()
        if filename.endswith(".svg"):
            # Optional: Convert SVG to PNG using cairosvg
            import cairosvg
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                cairosvg.svg2png(bytestring=contents, write_to=tmp.name)
                temp_path = tmp.name
        else:
            # Handle common raster images
            try:
                image = Image.open(io.BytesIO(contents)).convert("RGB")
            except UnidentifiedImageError:
                raise HTTPException(status_code=400, detail="Unsupported image format.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image.save(tmp.name, format="PNG")
                temp_path = tmp.name

        # Run sync function
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, generate_im2im_task, prompt, temp_path, seed)

        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        image_path = result["image_path"]

        if return_base64:
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            return JSONResponse(content={
                "status": "success",
                "image_base64": f"data:image/png;base64,{encoded_string}",
                "seed": result["seed"]
            })

        return FileResponse(
            path=image_path,
            media_type="image/png",
            filename=os.path.basename(image_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
