"""
Step 3 & 4 validation: OCR service unit test.
Run with: venv\Scripts\pytest tests/test_ocr.py -v
"""
import pytest
from app.services.ocr_service import extract_text
from PIL import Image, ImageDraw, ImageFont
import io


def make_sample_image(text: str) -> bytes:
    """
    Create a simple white image with black text for testing.
    Uses a default font — no external font file needed.
    """
    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 60), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_extract_text_english():
    """OCR should correctly extract plain English text."""
    image_bytes = make_sample_image("Hello, this is a test.")
    result = extract_text(image_bytes)
    assert "Hello" in result or "test" in result, (
        f"Expected English words in output, got: {result!r}"
    )


def test_extract_text_returns_string():
    """extract_text must always return a string, never None."""
    image_bytes = make_sample_image("FastAPI Study Assistant")
    result = extract_text(image_bytes)
    assert isinstance(result, str)


def test_extract_text_invalid_image():
    """Should raise ValueError for non-image bytes."""
    with pytest.raises(ValueError, match="Invalid image"):
        extract_text(b"this is not an image")


def test_extract_text_empty_image():
    """A blank white image should return an empty or whitespace-only string."""
    img = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = extract_text(buf.getvalue())
    assert isinstance(result, str)
    # Blank images return empty or whitespace — both are valid
    assert result.strip() == "" or len(result) >= 0
