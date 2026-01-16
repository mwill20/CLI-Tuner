"""
Response schemas for CLI-Tuner output.
"""
from pydantic import BaseModel
from typing import Literal, Optional

class ValidatedCommand(BaseModel):
    """Validated command after guardrails."""
    command: str
    risk_level: Literal["safe", "review", "dangerous"]
    confidence: float
    validation_results: dict

class CommandResponse(BaseModel):
    """Final response to user."""
    command: str
    risk_level: Literal["safe", "review", "dangerous"]
    confidence: float
    warning: Optional[str] = None
    provenance: dict

    class Config:
        frozen = True  # Immutable
