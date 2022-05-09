from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import logging
import pandas as pd
import numpy as np
from pickle import dump
from sklearn.compose import ColumnTransformer


APPLICATION_NAME = 'ML_project'
logger = logging.getLogger(APPLICATION_NAME)


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """
    Custom Standard Scaler
    """

    def __init__(self, copy=True):
        self.mean = None
        self.std = None
        self.copy = copy

    def fit(self, X):
        """ Fit transformer"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        logger.info(f"Standard scaler is being fitted, mean is {self.mean} and std {self.std}")
        return self

    def transform(self, X):
        """Transform data with mean and standard deviation"""
        X -= self.mean
        X /= self.std
        return X


class DataPreprocessor:
    """ Data processing pipeline to process categorical and numerical feature """
    def __init__(self, categorical_features: List[str], numerical_features: List[str],):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.pipeline = None

    def fit(self, data):
        numerical_transformer = CustomStandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )
        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor)]
        )
        self.pipeline.fit(data)

    def transform(self, data: pd.DataFrame) -> np.array:
        if not data.empty:
            return pd.DataFrame(self.pipeline.transform(data))
        return np.array([])

    def save_transformer(self, path: str) -> None:
        with open(path, 'wb') as f:
            dump(self.pipeline, f)
