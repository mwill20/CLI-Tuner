"""
Provenance tracking schemas.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DataProvenance(BaseModel):
    """Data source provenance."""
    source: str
    source_url: Optional[str]
    collected_date: datetime
    validated: bool
    validation_method: str

class ModelProvenance(BaseModel):
    """Model training provenance."""
    base_model: str
    training_data_hash: str
    config_hash: str
    trained_by: str
    trained_at: datetime
    wandb_run_id: str
    git_commit: str

class GenerationProvenance(BaseModel):
    """Generation-time provenance."""
    request_id: str
    model_version: str
    timestamp: datetime
    user_id_hash: str
    query_hash: str
    command_generated: str
    validation_passed: bool
