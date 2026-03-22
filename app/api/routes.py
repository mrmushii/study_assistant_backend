from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
from app.models.schemas import ProcessResponse
from app.services.ai_service import generate_study_content_from_image

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/process-file", response_model=ProcessResponse)
async def process_file(file: UploadFile = File(...)):
    """
    POST /process-file
    Accepts an image file and routes it directly to the local Vision LLM 
    to extract text, understand handwriting, and generate study content.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    try:
        # Step 1: Read the raw image bytes
        image_bytes = await file.read()
        
        # Step 2: Use Vision LLM to immediately extract and structure knowledge
        final_data = generate_study_content_from_image(image_bytes)
        
        return ProcessResponse(**final_data)
        
    except ValueError as ve:
        # Expected errors like Invalid Image or parse failure
        logger.warning(f"Validation error in process_file: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        # Errors from Ollama offline or failing
        logger.error(f"Runtime generation error: {str(re)}")
        raise HTTPException(status_code=500, detail="Internal AI processing error. Please check logs.")
    except Exception as e:
        logger.error(f"Unexpected error matching process_file end-to-end: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

