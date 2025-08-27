# config_annotations.py
from dataclasses import dataclass, field
from typing import List
@dataclass
class DatasetConfig:
    path: dict[str, str]

@dataclass
class PreprocessingConfig:
    steps: List[str] = field(default_factory=list)

@dataclass
class ModelConfig:
    path: dict[str, str]

@dataclass
class EvaluationConfig:
    metrics: dict[str, List[str]]

@dataclass
class MainConfig:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    evaluation: EvaluationConfig

