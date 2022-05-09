import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
from typing import List, Optional


@dataclass()
class PathParams:
    input_data_path: str
    output_path: str


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str]


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=17)


@dataclass()
class RandomForestParams:
    n_estimators: int = field(default=100)
    random_state: int = field(default=17)
    max_depth: int = field(default=3)
    criterion: str = field(default='gini')


@dataclass()
class TrainPipelineParams:
    """ Defines train pipeline parameters. """
    path_config: PathParams
    split_params: SplittingParams
    feature_params: FeatureParams
    train_params: RandomForestParams



TrainPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_training_pipeline_params(path: str) -> TrainPipelineParamsSchema:
    with open(path, "r") as input_stream:
        schema = TrainPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
