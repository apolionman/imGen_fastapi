from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Dict, Optional
import tempfile, os, httpx, asyncio, subprocess, whisper
from uuid import uuid4
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}