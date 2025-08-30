from dataclasses import dataclass, field
from typing import List

@dataclass
class PathConfig:
    path: dict[str, str]

@dataclass
class DatasetConfig:
    path: PathConfig
    features: dict[str, List[str]]

@dataclass
class PreprocessingConfig:
    random_state: int
    test_size: float
    feature_store: dict[str, list[str] | str]
    steps: dict[str, dict[str]]
    features: dict[str, list[str] | str]

@dataclass
class ModelConfig:
    path: dict[str, str] = PathConfig

@dataclass
class EvaluationConfig:
    metrics: dict[str, List[str]]

@dataclass
class MainConfig:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    pipeline: str

