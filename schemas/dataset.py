"""
Dataset schemas for data pipeline validation.
"""
from pydantic import BaseModel, Field


class BashCommandExample(BaseModel):
    """Schema for raw dataset examples."""

    instruction: str = Field(min_length=3, max_length=500)
    input: str = Field(default="", max_length=200)
    output: str = Field(min_length=1, max_length=500)
