from typing import List

import pytest
import pandas as pd
from faker import Faker
from src.entities import params
from sklearn.ensemble import RandomForestClassifier


ROWS = 200


@pytest.fixture(scope="session")
def input_data_path() -> str:
    return "..src/data/heart_cleveland_upload.csv"


@pytest.fixture(scope="session")
def output_path() -> str:
    return "models"


@pytest.fixture(scope="session")
def path_params(input_data_path, output_path):
    pathes = params.PathParams(
        input_data_path=input_data_path,
        output_path=output_path
    )
    return pathes


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope="session")
def target_col() -> str:
    return "condition"


@pytest.fixture(scope="session")
def feature_params(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str):
    feature_params = params.FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    return feature_params


@pytest.fixture(scope="session")
def fake_data() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(42)
    fake_data = {
        "age": [fake.pyint(min_value=30, max_value=80) for _ in range(ROWS)],
        "sex": [fake.pyint(min_value=0, max_value=1) for _ in range(ROWS)],
        "cp": [fake.pyint(min_value=0, max_value=3) for _ in range(ROWS)],
        "trestbps": [fake.pyint(min_value=94, max_value=200) for _ in range(ROWS)],
        "chol": [fake.pyint(min_value=126, max_value=555) for _ in range(ROWS)],
        "fbs": [fake.pyint(min_value=0, max_value=1) for _ in range(ROWS)],
        "restecg": [fake.pyint(min_value=0, max_value=2) for _ in range(ROWS)],
        "thalach": [fake.pyint(min_value=71, max_value=202) for _ in range(ROWS)],
        "exang": [fake.pyint(min_value=0, max_value=1) for _ in range(ROWS)],
        "oldpeak": [fake.pyfloat(min_value=0, max_value=7) for _ in range(ROWS)],
        "slope": [fake.pyint(min_value=0, max_value=2) for _ in range(ROWS)],
        "ca": [fake.pyint(min_value=0, max_value=4) for _ in range(ROWS)],
        "thal": [fake.pyint(min_value=0, max_value=3) for _ in range(ROWS)],
        "condition": [fake.pyint(min_value=0, max_value=1) for _ in range(ROWS)]
    }

    return pd.DataFrame(data=fake_data)


@pytest.fixture(scope="session")
def random_forest_training_params():
    model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=4,
        random_state=17
    )
    return model


@pytest.fixture(scope="session")
def splitting_params():
    split_params = params.SplittingParams(
        val_size=0.2,
        random_state=42)
    return split_params


@pytest.fixture(scope="session")
def config(path_params, splitting_params, feature_params, random_forest_training_params):
    mock_config = params.TrainPipelineParams(
        path_config=path_params,
        split_params=splitting_params,
        feature_params=feature_params,
        train_params=random_forest_training_params,

    )
    return mock_config