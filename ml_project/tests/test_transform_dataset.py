from src.data.transform_dataset import CustomStandardScaler, DataPreprocessor
from src.entities import params
import numpy as np
import pandas as pd
import pytest
from typing import List


ROWS = 100

test_X = np.array([[10, 118, 47], [-1, -12, -45], [0.0, 0.1, 0.2]])
expected_X = test_X.copy()
expected_X -= np.mean(expected_X, axis=0)
expected_X /= np.std(expected_X, axis=0)

@pytest.mark.parametrize(
    "test_input, expected", [(test_X, expected_X)]
)
def test_CustomStandardScaler(
    test_input: np.ndarray, expected: np.ndarray
) -> None:
    transformer = CustomStandardScaler()
    transformer.fit(test_input)
    scaled_input = transformer.transform(test_input)

    assert (
            scaled_input.tolist() == expected.tolist()
        ), "CustomStandardScaler failed to process data"


def test_DataPreprocessor(fake_data: pd.DataFrame,
                          numerical_features: List,
                          categorical_features: List,
                          target_col: str):
    transformer = DataPreprocessor(categorical_features, numerical_features)
    transformer.fit(fake_data.drop("condition", axis=1))
    transformed_data = transformer.transform(fake_data.drop(target_col, axis=1))
    std_scaled = transformed_data.iloc[:, :len(numerical_features)]
    scaler = CustomStandardScaler()
    scaler.fit(fake_data[numerical_features])
    expected = scaler.transform(fake_data[numerical_features])

    assert np.allclose(std_scaled, expected)
    assert (fake_data.shape[0], 30) == transformed_data.shape
