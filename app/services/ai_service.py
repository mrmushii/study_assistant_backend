import os
import json
from ollama import Client
from typing import Dict, Any

# Load config from .env (populated in Step 1)
_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Initialize the Ollama client
client = Client(host=_OLLAMA_URL)


def generate_study_content(cleaned_text: str) -> Dict[str, Any]:
    """
    [STUB]
    Calls the local Ollama LLM to generate explanation, summary, 
    key points, and questions based on the cleaned OCR text.
    
    Will be fully implemented with the prompt in Step 7.
    
    Returns:
        A dictionary matching the ProcessResponse schema.
    """
    # Just a placeholder for Step 5 validation
    preview = cleaned_text[:30] + "..." if len(cleaned_text) > 30 else cleaned_text
    return {
        "explanation": f"[Stub] AI Explanation based on text: {preview}",
        "summary": "[Stub] AI Summary",
        "key_points": ["[Stub] AI Key Point 1", "[Stub] AI Key Point 2"],
        "questions": ["[Stub] AI Question 1", "[Stub] AI Question 2"]
    }
