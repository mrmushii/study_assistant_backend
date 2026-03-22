import os
import json
from ollama import Client
from typing import Dict, Any

from app.models.schemas import ProcessResponse
import logging

# Load config from .env (populated in Step 1)
_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Initialize the Ollama client
_client = Client(host=_OLLAMA_URL)

logger = logging.getLogger(__name__)

STUDY_PROMPT_TEMPLATE = """You are an expert tutor for a university student in Bangladesh.
Your goal is to explain the provided study material clearly and simply.

Please analyze the following text and provide:
1. An explanation (Explain simply using a mix of Bangla and English)
2. A summary (A short overview of the entire text)
3. Key points (Extract the most important technical or conceptual points)
4. 5 questions (Generate exactly 5 questions to test the student's understanding)

Text to analyze:
{cleaned_text}

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

def generate_study_content(cleaned_text: str) -> dict:
    """
    Calls the local Ollama LLM to generate explanation, summary, 
    key points, and questions based on the cleaned OCR text.
    
    Args:
        cleaned_text: Cleaned text from Step 6.
        
    Returns:
        A dictionary matching the ProcessResponse schema.
    """
    if not cleaned_text or not cleaned_text.strip():
        # Fallback for completely empty document
        return {
            "explanation": "No text could be extracted from the image.",
            "summary": "Empty Document",
            "key_points": [],
            "questions": []
        }

    prompt = STUDY_PROMPT_TEMPLATE.format(cleaned_text=cleaned_text)
    
    try:
        response = _client.generate(
            model=_OLLAMA_MODEL,
            prompt=prompt,
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
        logger.error(f"Failed to parse LLM JSON: {e}")
        raise ValueError("AI produced invalid JSON output. Please try again.")
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        raise RuntimeError(f"AI generation failed: {str(e)}")

