from fastapi import APIRouter, UploadFile, File
import tempfile
import shutil
import os

from app.services.audio_analysis import analyze_audio

router = APIRouter()

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        result = analyze_audio(temp_path)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)