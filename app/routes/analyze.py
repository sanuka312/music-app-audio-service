import os
import tempfile
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.audio_analysis import analyze_audio

router = APIRouter()

@router.post("/analyze")
async def analyze(
    file: Optional[UploadFile] = File(default=None),
    filePath: Optional[str] = Form(default=None),
):
    temp_path = None
    try:
        if file is not None:
            suffix = os.path.splitext(file.filename or "")[1] or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                temp_path = tmp.name
            return analyze_audio(temp_path)

        if filePath:
            if not os.path.exists(filePath):
                raise HTTPException(status_code=400, detail="Provided filePath does not exist")
            return analyze_audio(filePath)

        raise HTTPException(
            status_code=400,
            detail="No audio input provided. Send multipart 'file' or form 'filePath'.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(exc)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)