"""
Step 6 validation: Text Cleaner unit test.
Run with: venv\Scripts\pytest tests/test_text_cleaner.py -v
"""
import pytest
from unittest.mock import patch
from app.services.text_cleaner import clean_ocr_text, CLEANING_PROMPT_TEMPLATE

def test_clean_ocr_text_empty():
    """Should return empty string immediately if input is empty."""
    assert clean_ocr_text("") == ""
    assert clean_ocr_text("   \n  ") == ""

@patch("app.services.text_cleaner._client.generate")
def test_clean_ocr_text_calls_llm(mock_generate):
    """Should construct the right prompt and call Ollama generation."""
    # Mock the response from Ollama
    mock_generate.return_value = {"response": "This is cleaned Bangla text."}
    
    noisy_text = "Ths iz a test banglish text."
    result = clean_ocr_text(noisy_text)
    
    # Verify the output is what we mocked
    assert result == "This is cleaned Bangla text."
    
    # Verify Ollama was called correctly
    mock_generate.assert_called_once()
    called_kwargs = mock_generate.call_args.kwargs
    
    # Verify the prompt contained the noisy text and the rules
    assert "Convert Banglish" in called_kwargs["prompt"]
    assert noisy_text in called_kwargs["prompt"]
    assert called_kwargs["stream"] is False
