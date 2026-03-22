from app.models.schemas import ProcessResponse
import logging
import os
import json
from ollama import Client

# Load config from .env
_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision")

# Initialize the Ollama client
_client = Client(host=_OLLAMA_URL)

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

def generate_study_content_from_image(image_bytes: bytes) -> dict:
    """
    Calls the local Ollama Vision LLM to generate explanation, summary, 
    key points, and questions directly from the raw image.
    
    Args:
        image_bytes: Raw bytes of the uploaded image.
        
    Returns:
        A dictionary matching the ProcessResponse schema.
    """
    if not image_bytes:
        raise ValueError("No image received.")

    try:
        response = _client.generate(
            model=_OLLAMA_MODEL,
            prompt=VISION_PROMPT_TEMPLATE,
            images=[image_bytes],
            format="json",
            stream=False
        )
        
        # Parse the output to ensure it's valid JSON
        result_text = response["response"].strip()
        parsed = json.loads(result_text)
        
        # Validate against our Pydantic schema to ensure all keys exist
        validated_data = ProcessResponse(**parsed).model_dump()
        return validated_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON from vision: {e}")
        raise ValueError("AI produced invalid JSON output. Please try again.")
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        raise RuntimeError(f"AI generation failed: {str(e)}")

