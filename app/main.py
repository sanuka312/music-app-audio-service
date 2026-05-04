from fastapi import FastAPI
from app.routes.analyze import router as analyze_router

app = FastAPI(title="Music Analyzer Python Service")

app.include_router(analyze_router)

@app.get("/health")
def health():
    return {
        "success": True,
        "message": "Python analysis service is running"
    }