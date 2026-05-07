from fastapi import APIRouter, UploadFile, File

from pydantic import BaseModel
from app.services.audio_analysis import analyze_audio

router = APIRouter()

class AnalysisRequest(BaseModel):
    filePath: str

@router.post("/analyze")
async def analyze(request: AnalysisRequest):
    result = analyze_audio(request.filePath)
    return result