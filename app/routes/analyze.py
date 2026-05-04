from fastapi import APIRouter
from pydantic import BaseModel
from app.services.audio_analysis import analyze_audio

router = APIRouter()

class AnalyzeRequest(BaseModel):
    filePath: str

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    result = analyze_audio(request.filePath)
    return result