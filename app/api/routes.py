from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
from app.models.schemas import ProcessResponse
from app.services.ocr_service import extract_text
from app.services.text_cleaner import clean_ocr_text
from app.services.ai_service import generate_study_content

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/process-file", response_model=ProcessResponse)
async def process_file(file: UploadFile = File(...)):
    """
    POST /process-file
    Accepts an image file, runs OCR, cleans text, and returns AI-generated study content.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    try:
        # Step 1: Read the image bytes
        image_bytes = await file.read()
        
        # Step 2: Extract raw text via Tesseract (OCR)
        raw_text = extract_text(image_bytes)
        
        # Step 3: Clean the recognized text (Fix Bangla, Banglish -> Bangla) via LLM
        cleaned_text = clean_ocr_text(raw_text)
        
        # Step 4: Generate explanations, summaries, and questions
        final_data = generate_study_content(cleaned_text)
        
        return ProcessResponse(**final_data)
        
    except ValueError as ve:
        # Expected errors like Invalid Image, or AI output failing schema parse
        logger.warning(f"Validation error in process_file: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        # Errors from dependencies like Tesseract missing or Ollama offline
        logger.error(f"Runtime extraction error: {str(re)}")
        raise HTTPException(status_code=500, detail="Internal processing error. Please check logs.")
    except Exception as e:
        logger.error(f"Unexpected error matching process_file end-to-end: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

