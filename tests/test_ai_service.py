"""
Step 7 validation: AI Service unit test.
Run with: venv\Scripts\pytest tests/test_ai_service.py -v
"""
import pytest
import json
from unittest.mock import patch
from app.services.ai_service import generate_study_content

def test_generate_study_content_empty():
    """Should return a fallback structure if the input text is empty."""
    result = generate_study_content("   ")
    assert result["explanation"] == "No text could be extracted from the image."
    assert result["summary"] == "Empty Document"
    assert result["key_points"] == []
    assert result["questions"] == []

@patch("app.services.ai_service._client.generate")
def test_generate_study_content_success(mock_generate):
    """Should call Ollama, enforce JSON format, and validate against Pydantic schema."""
    
    # Mock a valid JSON response from Ollama
    mock_llm_output = {
        "explanation": "This text explains Python decorators briefly.",
        "summary": "Python decorators wrap functions.",
        "key_points": ["Decorators start with @", "They modify logic"],
        "questions": ["What is a decorator?", "How do you use it?", "Q3", "Q4", "Q5"]
    }
    
    # The SDK returns a dict with 'response' as a string containing the text
    mock_generate.return_value = {"response": json.dumps(mock_llm_output)}
    
    result = generate_study_content("Decorators modifying functions @wrapper")
    
    # Verify the output matches our mock after passing through Pydantic validation
    assert result == mock_llm_output
    
    # Verify Ollama was called correctly
    mock_generate.assert_called_once()
    called_kwargs = mock_generate.call_args.kwargs
    assert called_kwargs["format"] == "json"  # Essential for structured output
    assert called_kwargs["stream"] is False

@patch("app.services.ai_service._client.generate")
def test_generate_study_content_invalid_json(mock_generate):
    """Should raise ValueError if the model returns garbage instead of JSON."""
    mock_generate.return_value = {"response": "I cannot help with that."}
    
    with pytest.raises(ValueError, match="invalid JSON output"):
        generate_study_content("Read this textbook page")

@patch("app.services.ai_service._client.generate")
def test_generate_study_content_missing_fields(mock_generate):
    """Should raise ValidationError if the model returns JSON but it misses required fields."""
    # Missing 'questions' and 'explanation'
    bad_json = {
        "summary": "Missing fields",
        "key_points": []
    }
    mock_generate.return_value = {"response": json.dumps(bad_json)}
    
    with pytest.raises(RuntimeError) as exc_info:
        generate_study_content("Text with missing fields")
        
    assert "explanation" in str(exc_info.value)
