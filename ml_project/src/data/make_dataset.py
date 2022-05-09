# -*- coding: utf-8 -*-
import logging
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.entities.params import SplittingParams


APPLICATION_NAME = 'ML_project'
logger = logging.getLogger(APPLICATION_NAME)


def read_csv(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    logger.info(f"Dataframe has shape: {data.shape}")
    return data


def get_train_test_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data to train and tests subsets"""
    train_set, test_set = train_test_split(data,
                                           test_size=SplittingParams.val_size,
                                           random_state=SplittingParams.random_state)
    logger.info(f"Train set has shape: {train_set.shape}, test set - {test_set.shape} ")
    return train_set, test_set
