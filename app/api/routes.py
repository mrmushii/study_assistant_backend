from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import logging
import json
import io
import fitz
from PIL import Image
from app.services.ai_service import generate_study_content_stream

logger = logging.getLogger(__name__)
router = APIRouter()

def resize_image_bytes(image_bytes: bytes, max_size: int = 1024) -> bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Image resize failed: {e}")
        return image_bytes

def stitch_images_vertically(image_bytes_list: list[bytes]) -> bytes:
    try:
        images = [Image.open(io.BytesIO(b)) for b in image_bytes_list]
        total_height = sum(img.height for img in images)
        max_w = max(img.width for img in images)
        
        stitched_img = Image.new('RGB', (max_w, total_height))
        y_offset = 0
        for img in images:
            stitched_img.paste(img, (0, y_offset))
            y_offset += img.height
            
        buf = io.BytesIO()
        stitched_img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Image stitching failed: {e}")
        return image_bytes_list[0]

@router.post("/process-file")
async def process_file(file: UploadFile = File(...)):
    """
    POST /process-file
    Accepts an image or PDF file.
    - If PDF: Extracts text cleanly. If scanned, extracts images, resizes, and uses Vision LLM.
    - If Image: Resizes the image and routes directly to the Vision LLM.
    Returns a Server-Sent Events (SSE) stream allowing the frontend to show 
    dynamic progress (TPS, ETA, %) before finally sending the extracted data.
    """
    if not (file.content_type.startswith("image/") or file.content_type == "application/pdf"):
        raise HTTPException(status_code=400, detail="Only image and PDF files are supported.")

    try:
        # Step 1: Read the raw image bytes
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")
        
    async def event_generator():
        try:
            content_type = file.content_type
            if content_type == "application/pdf":
                doc = fitz.open(stream=image_bytes, filetype="pdf")
                pdf_text = ""
                images_to_process = []
                for page in doc:
                    text = page.get_text()
                    if len(text.strip()) > 50:
                        pdf_text += text + "\n"
                    else:
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        images_to_process.append(resize_image_bytes(img_bytes))
                
                if images_to_process:
                    if len(images_to_process) > 1:
                        if len(images_to_process) > 4:
                            yield f"data: {json.dumps({'status': 'error', 'message': 'PDF contains too many scanned pages. Maximum 4 scanned pages supported for handwriting analysis.'})}\n\n"
                            return
                        stitched_img_bytes = stitch_images_vertically(images_to_process)
                        async for event_data in generate_study_content_stream(images=[stitched_img_bytes]):
                            yield f"data: {json.dumps(event_data)}\n\n"
                    else:
                        async for event_data in generate_study_content_stream(images=images_to_process):
                            yield f"data: {json.dumps(event_data)}\n\n"
                elif pdf_text.strip():
                    async for event_data in generate_study_content_stream(text=pdf_text.strip()):
                        yield f"data: {json.dumps(event_data)}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'No text or image could be extracted.'})}\n\n"
            else:
                # Target is an image
                resized_bytes = resize_image_bytes(image_bytes)
                async for event_data in generate_study_content_stream(images=[resized_bytes]):
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

