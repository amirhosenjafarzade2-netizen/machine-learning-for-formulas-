# config.py
from pydantic import BaseModel, Field, field_validator
import numpy as np

class Config(BaseModel):
    """Configuration model with validation."""
    random_seed: int = 42
    min_rows: int = Field(default=10, gt=0)
    epochs: int = Field(default=50, gt=0)
    validation_split: float = Field(default=0.2, ge=0.0, le=1.0)
    bubble_point_default: float = 2000
    pop_size: int = Field(default=50, gt=0)
    n_generations: int = Field(default=50, gt=0)
    n_calls: int = Field(default=50, gt=0)
    visualizations: list[str] = Field(default_factory=lambda: [
        "correlation", "mi", "mi_saturated", "mi_undersaturated", "pairplot", 
        "parallel", "boxplot", "pca", "pca_3d", "rf_importance", "shap"
    ])
    nn_layers: list[int] = Field(default_factory=lambda: [64, 32])
    nn_activation: str = "relu"
    batch_size: int = Field(default=32, gt=0)
    n_jobs: int = Field(default=-1, ge=-1)
    shap_sample_size: int = Field(default=100, gt=0)

    @field_validator('visualizations')
    @classmethod
    def validate_visualizations(cls, v):
        valid = ["correlation", "mi", "mi_saturated", "mi_undersaturated", "pairplot", 
                 "parallel", "boxplot", "pca", "pca_3d", "rf_importance", "shap"]
        if not all(x in valid for x in v):
            raise ValueError(f"Invalid visualization types: {set(v) - set(valid)}")
        return v
