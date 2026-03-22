# AI Study Assistant

An offline, modular AI system for students — designed for low-resource environments.

## Features (Phased)
- **Phase 1 (Now):** OCR → Text Cleaning → AI Explanation/Summary/Q&A via local LLM
- **Phase 2 (Later):** Question Analyzer, Math Solver, Voice features

## Stack
| Layer | Tool |
|---|---|
| API | FastAPI |
| OCR | Tesseract (ben+eng) |
| LLM | Ollama (local, offline) |
| Language | Python 3.10+ |

## Project Structure
```
app/
  main.py          # FastAPI entry point
  api/routes.py    # HTTP endpoints
  services/        # Business logic (OCR, cleaning, AI)
  models/          # Pydantic schemas
tests/             # Step-by-step test files
```

## Setup & Running

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start the dev server
python -m uvicorn app.main:app --reload --port 8000
```

## API
| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Server health check |
| POST | /process-file | Upload image → get study content |
