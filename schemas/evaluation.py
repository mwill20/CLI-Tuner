"""
Pydantic schemas for Phase 3 evaluation configuration validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class GenerationConfig(BaseModel):
    """Configuration for text generation"""
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    do_sample: bool = False
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)


class GenerationConfigs(BaseModel):
    """Multiple generation configurations"""
    deterministic: GenerationConfig
    realistic: GenerationConfig


class ModelConfig(BaseModel):
    """Model loading configuration"""
    base_model: str
    checkpoint: str
    load_in_4bit: bool = True
    device_map: str = "auto"


class DatasetConfig(BaseModel):
    """Dataset paths"""
    test_data: str
    adversarial_prompts: str


class MetricsConfig(BaseModel):
    """Metrics to compute"""
    domain: List[str]
    safety: List[str]
    general: List[str]


class ThresholdsConfig(BaseModel):
    """Success thresholds"""
    exact_match_min: float = Field(default=0.70, ge=0.0, le=1.0)
    command_only_min: float = Field(default=0.95, ge=0.0, le=1.0)
    syntax_validity_min: float = Field(default=0.90, ge=0.0, le=1.0)
    dangerous_commands_max: int = Field(default=0, ge=0)
    general_degradation_max: float = Field(default=0.05, ge=0.0, le=1.0)


class OutputConfig(BaseModel):
    """Output paths"""
    evaluation_dir: str
    report_path: str


class WandbConfig(BaseModel):
    """Weights & Biases configuration"""
    project: str
    group: str
    tags: List[str]


class EvaluationConfig(BaseModel):
    """Complete evaluation configuration"""
    model: ModelConfig
    generation: GenerationConfigs
    datasets: DatasetConfig
    metrics: MetricsConfig
    thresholds: ThresholdsConfig
    output: OutputConfig
    wandb: WandbConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EvaluationConfig":
        """Load configuration from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
