# Music App Audio Service

Small FastAPI service for audio upload and analysis.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Analyze: `POST http://localhost:8000/analyze` with multipart form field `file` (audio)

API docs: `http://localhost:8000/docs`

## Layout

- `app/main.py` — FastAPI app and routers
- `app/routes/analyze.py` — analyze endpoints
- `app/services/audio_analysis.py` — analysis logic
