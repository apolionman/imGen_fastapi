from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import asyncio, os, base64
from app.scripts.flux_run import generate_image_task

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
