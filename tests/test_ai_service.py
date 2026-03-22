"""
Step 10 validation: AI Service Streaming unit test.
Run with: venv\Scripts\pytest tests/test_ai_service.py -v
"""
import pytest
import json
from unittest.mock import AsyncMock, patch
from app.services.ai_service import generate_study_content_stream

@pytest.mark.asyncio
async def test_generate_study_content_empty():
    """Should return an error event if input is empty."""
    events = [e async for e in generate_study_content_stream(b"")]
    assert len(events) == 1
    assert events[0]["status"] == "error"
    assert "No image received." in events[0]["message"]

@pytest.mark.asyncio
@patch("app.services.ai_service._client.generate", new_callable=AsyncMock)
async def test_generate_study_content_success(mock_generate):
    """Should stream expected progress events and finally complete with data."""
    
    mock_llm_output = {
        "explanation": "Test explanation.",
        "summary": "Test summary.",
        "key_points": ["Point 1"],
        "questions": ["Q1", "Q2", "Q3", "Q4", "Q5"]
    }
    
    # Simulate an async iterable returning chunks
    async def mock_stream_response():
        yield {"response": json.dumps(mock_llm_output)}
        
    mock_generate.return_value = mock_stream_response()
    
    events = [event async for event in generate_study_content_stream(b"mock_image_bytes")]
    
    # Evaluate the events
    assert events[0]["status"] == "starting"
    assert events[1]["status"] == "processing"
    
    # After the stream yields the fast single chunk, it finalizes
    assert events[-2]["status"] == "finalizing"
    assert events[-1]["status"] == "complete"
    assert events[-1]["data"] == mock_llm_output

@pytest.mark.asyncio
@patch("app.services.ai_service._client.generate", new_callable=AsyncMock)
async def test_generate_study_content_invalid_json(mock_generate):
    """Should yield an error event if LLM returns garbage."""
    async def mock_stream_response():
        yield {"response": "I am an AI and I cannot do that"}
        
    mock_generate.return_value = mock_stream_response()
    
    events = [e async for e in generate_study_content_stream(b"mock_bytes")]
    
    assert events[-1]["status"] == "error"
    assert "invalid JSON" in events[-1]["message"]
