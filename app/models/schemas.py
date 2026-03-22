from pydantic import BaseModel
from typing import List


class ProcessResponse(BaseModel):
    """
    The standard response format for POST /process-file.
    All fields are required and must be populated by the AI service.
    """
    explanation: str
    summary: str
    key_points: List[str]
    questions: List[str]
