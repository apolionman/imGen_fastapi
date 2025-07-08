from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
import os, json, uuid, shutil, tempfile, httpx, re, asyncio, requests
from uuid import UUID
import os
from huggingface_hub import login

login(token=os.getenv("HUGGINGFACE_TOKEN"))

# routes
from app.routes.endpoints import router as endpoints_router

app = FastAPI()

app.mount("/generated/images", StaticFiles(directory="/app/output"), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints_router, prefix="/api/v1", tags=["Backend Endpoints"])