from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from dataclasses import dataclass
from typing import List

from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from typing import Union

@dataclass
class ModelInstance:
    # Using field(default_factory) to avoid mutable default arguments
    random_forest: RandomForestClassifier = field(default_factory=RandomForestClassifier)
    svc: SVC = field(default_factory=SVC)
    knn: KNeighborsClassifier = field(default_factory=KNeighborsClassifier)
    
    def get_instance(self, model_name: str) -> Union[RandomForestClassifier, SVC, KNeighborsClassifier, None]:

        model_mapping = {
            'random_forest': self.random_forest,
            'svc': self.svc,
            'knn': self.knn
        }
        
        return model_mapping.get(model_name)


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
    path: PathConfig
    passed_model: str
    list_of_models: list[str]
    models: dict[str, dict[str, list[str]]]

@dataclass
class EvaluationConfig:
    metrics: dict[str, List[str]]

@dataclass
class PipelineConfig:
    type: str
    local: bool

@dataclass
class MainConfig:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    pipeline: PipelineConfig

