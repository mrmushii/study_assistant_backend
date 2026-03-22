from app.models.schemas import ProcessResponse
import logging
import os
import json
import time
from ollama import AsyncClient

# Load config from .env
_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision")

# Initialize the async Ollama client
_client = AsyncClient(host=_OLLAMA_URL)

logger = logging.getLogger(__name__)

VISION_PROMPT_TEMPLATE = """You are an expert tutor for a university student in Bangladesh.
Your goal is to explain the handwritten or printed study material shown in this image clearly and simply.

Please analyze the text in the image and provide:
1. An explanation (Explain simply using a mix of Bangla and English)
2. A summary (A short overview of the entire text)
3. Key points (Extract the most important technical or conceptual points)
4. 5 questions (Generate exactly 5 questions to test the student's understanding)

OUTPUT INSTRUCTIONS:
You MUST respond with valid raw JSON only. Do not include markdown code blocks (like ```json).
Do not include any greeting or conversational text before or after the JSON.
Your JSON must exactly match this structure:
{{
    "explanation": "Your explanation here (Bangla + English mix)",
    "summary": "Your summary here",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "questions": ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]
}}
"""

async def generate_study_content_stream(image_bytes: bytes):
    """
    Async generator that calls the local Ollama Vision LLM with stream=True.
    Yields progress tracking events before finally yielding the JSON data.
    """
    if not image_bytes:
        yield {"status": "error", "message": "No image received."}
        return

    yield {"status": "starting", "progress": 5, "message": "Initializing AI Vision analysis..."}
    
    try:
        start_time = time.time()
        
        response_stream = await _client.generate(
            model=_OLLAMA_MODEL,
            prompt=VISION_PROMPT_TEMPLATE,
            images=[image_bytes],
            format="json",
            stream=True
        )
        
        yield {"status": "processing", "progress": 10, "message": "Reading image and drafting response..."}
        
        full_text = ""
        token_count = 0
        # Average length of our study JSON is around 600-800 tokens. 
        # We use 700 as a baseline for percentage calculations.
        target_tokens = 700 
        
        async for chunk in response_stream:
            if 'response' in chunk:
                full_text += chunk['response']
                token_count += 1
                
                # Send an update every 5 tokens to keep it smooth but not too chatty
                if token_count % 5 == 0:
                    elapsed = time.time() - start_time
                    tps = token_count / elapsed if elapsed > 0 else 1
                    
                    remaining_tokens = max(50, target_tokens - token_count)
                    eta_seconds = remaining_tokens / tps
                    
                    # Progress scales from 10% to 95% based on token completion
                    dynamic_pct = min(95, 10 + int((token_count / target_tokens) * 85))
                    
                    yield {
                        "status": "processing", 
                        "progress": dynamic_pct, 
                        "eta_seconds": round(eta_seconds, 1),
                        "tokens_per_second": round(tps, 1)
                    }
                    
        # Finished streaming tokens
        yield {"status": "finalizing", "progress": 98, "message": "Validating AI response format..."}
        
        parsed = json.loads(full_text.strip())
        validated_data = ProcessResponse(**parsed).model_dump()
        
        yield {
            "status": "complete",
            "progress": 100,
            "data": validated_data
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON from vision: {e}")
        yield {"status": "error", "message": "AI produced invalid JSON output. Please try again."}
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        yield {"status": "error", "message": f"AI generation failed: {str(e)}"}

