import os
from ollama import Client

_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

_client = Client(host=_OLLAMA_URL)

CLEANING_PROMPT_TEMPLATE = """You are an expert editor for Bengali and English text.
Your task is to take the following noisy text extracted from an OCR engine and clean it.

Rules:
1. Fix any Bangla spelling mistakes.
2. Convert Banglish (Bengali written in English letters) into proper Bangla script.
3. Keep technical English terms in English.
4. Structure the text properly into readable paragraphs or bullet points.
5. Return ONLY the cleaned text. Do not add any conversational filler, explanations, or quotes.

Text to clean:
{raw_text}
"""

def clean_ocr_text(raw_text: str) -> str:
    """
    Cleans noisy OCR text using the local LLM.
    Fixes Bangla spelling, converts Banglish to Bangla, and preserves technical English.
    
    Args:
        raw_text: The noisy text string from Tesseract.
        
    Returns:
        The cleaned and properly formatted string.
    """
    if not raw_text or not raw_text.strip():
        return ""

    prompt = CLEANING_PROMPT_TEMPLATE.format(raw_text=raw_text)
    
    response = _client.generate(
        model=_OLLAMA_MODEL,
        prompt=prompt,
        stream=False
    )
    
    return response["response"].strip()
