"""
Request schemas for CLI-Tuner input validation.
"""
from pydantic import BaseModel, Field
from typing import Literal

class CommandRequest(BaseModel):
    """User request schema."""
    query: str = Field(min_length=3, max_length=500)
    shell_type: Literal["bash"] = "bash"
    safety_level: Literal["strict", "moderate"] = "strict"

class NormalizedRequest(BaseModel):
    """Normalized request after preprocessing."""
    prompt: str
    original_query: str
    metadata: dict
