import os
import pytesseract
from PIL import Image, UnidentifiedImageError
import io
import fitz  # PyMuPDF

# Read Tesseract binary path from environment variable.
# Falls back to the default Windows install path if not set.
_TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

# Use Bengali + English together.
# ben+eng handles mixed Bangla/Banglish/English study content.
_OCR_LANG = "ben+eng"

# Page segmentation mode 3 = fully automatic (best for full-page scans).
_OCR_CONFIG = "--psm 3"


def extract_text(image_bytes: bytes) -> str:
    """
    Extract text from raw image bytes using Tesseract OCR.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        Extracted text as a string. May be noisy — cleaning is done
        in text_cleaner.py (Step 6).

    Raises:
        ValueError: If the bytes cannot be decoded as a valid image.
        RuntimeError: If Tesseract fails to run.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise ValueError("Invalid image: could not decode the uploaded file.")

    try:
        text = pytesseract.image_to_string(
            image,
            lang=_OCR_LANG,
            config=_OCR_CONFIG,
        )
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            f"Tesseract not found at: {_TESSERACT_CMD}. "
            "Check TESSERACT_CMD in your .env file."
        )

    return text.strip()

def extract_text_from_file(file_bytes: bytes, content_type: str) -> str:
    """
    Extract text handling both PDFs and Images.
    If PDF, tries to extract text natively using PyMuPDF. If the page lacks text
    (e.g., scanned PDF), renders the page to an image and uses Tesseract OCR.
    """
    text = ""
    if content_type == "application/pdf":
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                page_text = page.get_text()
                if len(page_text.strip()) > 50:
                    text += page_text + "\n"
                else:
                    # Render page to image and OCR
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    text += extract_text(img_bytes) + "\n"
        except Exception as e:
            raise ValueError(f"Failed to process PDF: {e}")
    elif content_type.startswith("image/"):
        text = extract_text(file_bytes)
    else:
        raise ValueError("Unsupported file type.")
    
    return text.strip()
