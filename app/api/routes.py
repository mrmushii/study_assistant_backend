from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import logging
import json
from app.services.ai_service import generate_study_content_stream

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/process-file")
async def process_file(file: UploadFile = File(...)):
    """
    POST /process-file
    Accepts an image file and routes it directly to the local Vision LLM.
    Returns a Server-Sent Events (SSE) stream allowing the frontend to show 
    dynamic progress (TPS, ETA, %) before finally sending the extracted data.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    try:
        # Step 1: Read the raw image bytes
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")
        
    async def event_generator():
        try:
            # Step 2: Use Vision LLM async generator to stream dynamic progress
            async for event_data in generate_study_content_stream(image_bytes):
                # Yield in valid Server-Sent-Events format
                yield f"data: {json.dumps(event_data)}\n\n"
                
        except ValueError as ve:
            logger.warning(f"Validation error in process_file stream: {str(ve)}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(ve)})}\n\n"
        except RuntimeError as re:
            logger.error(f"Runtime generation error: {str(re)}")
            yield f"data: {json.dumps({'status': 'error', 'message': 'Internal AI processing error. Please check logs.'})}\n\n"
        except Exception as e:
            logger.error(f"Unexpected error matching process_file stream: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'message': 'An unexpected error occurred.'})}\n\n"

    # Return the streaming response instead of a static JSON response
    return StreamingResponse(event_generator(), media_type="text/event-stream")

